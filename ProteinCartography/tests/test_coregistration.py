"""Tests for the co-registration index.

The thing under test is a guarantee, not a calculation: that two spaces being
compared are over the same proteins, and that anything lost on the way to that
is named. So most of these tests are about what the report *says* rather than
about the intersection itself, which is three lines.

The failure this prevents has no symptom. Two spaces built over sets differing
by a handful of proteins reduce cleanly, plot cleanly, and produce a
per-protein comparison silently conditioned on the overlap. Nothing downstream
can notice, which is why the check has to be here.
"""

import numpy as np
import pytest
from coregistration import (
    CoregistrationError,
    compare_pair,
    k_nearest,
    neighborhood_jaccard,
    pairwise_distances,
    procrustes_disparity,
    rank_correlation,
    shared_index,
)
from index import ProteinIndex


def test_identical_sets_are_reported_as_exact():
    """The healthy case, and the one the demo produces."""
    report = shared_index({"a": ["P1", "P2", "P3"], "b": ["P1", "P2", "P3"]})
    assert report.index.as_list == ["P1", "P2", "P3"]
    assert report.is_exact
    assert report.n_shared == 3
    assert "every space had the full set" in report.describe()


def test_the_shared_index_follows_the_reference_order():
    """Order must come from somewhere nameable, not from set iteration."""
    report = shared_index({"a": ["P3", "P1", "P2"], "b": ["P1", "P2", "P3"]}, reference="a")
    assert report.index.as_list == ["P3", "P1", "P2"]

    other = shared_index({"a": ["P3", "P1", "P2"], "b": ["P1", "P2", "P3"]}, reference="b")
    assert other.index.as_list == ["P1", "P2", "P3"]


def test_the_reference_defaults_to_the_first_space():
    report = shared_index({"a": ["P2", "P1"], "b": ["P1", "P2"]})
    assert report.reference == "a"
    assert report.index.as_list == ["P2", "P1"]


def test_a_space_that_measured_extra_proteins_has_them_named():
    """The whole point: what falls out of the comparison is listed, not counted."""
    report = shared_index({"a": ["P1", "P2", "P3"], "b": ["P1", "P2"]})
    assert report.index.as_list == ["P1", "P2"]
    assert not report.is_exact
    assert report.contribution("a").dropped == ("P3",)
    assert report.contribution("b").dropped == ()
    assert report.contribution("a").n_own == 3


def test_the_description_names_the_dropped_proteins():
    report = shared_index({"a": ["P1", "P2", "P3"], "b": ["P1"]})
    described = report.describe()
    assert "conditioned on the overlap" in described
    assert "P2" in described and "P3" in described


def test_a_long_dropped_list_is_truncated_with_a_count():
    """A thousand-protein divergence must not produce a thousand-line warning."""
    report = shared_index({"a": [f"P{i}" for i in range(20)], "b": ["P0"]})
    assert "+14 more" in report.describe()
    assert report.contribution("a").n_dropped == 19


def test_three_spaces_intersect_pairwise_down_to_the_common_set():
    report = shared_index(
        {"a": ["P1", "P2", "P3"], "b": ["P2", "P3", "P4"], "c": ["P3", "P2", "P9"]}
    )
    assert report.index.as_list == ["P2", "P3"]
    assert report.contribution("c").dropped == ("P9",)


def test_disjoint_spaces_are_an_error_that_names_both():
    """An empty intersection is not a small overlap; there is nothing to compare."""
    with pytest.raises(CoregistrationError, match="share no proteins"):
        shared_index({"structure": ["P1"], "families": ["P2"]})


def test_a_duplicate_protid_in_one_space_is_an_error():
    """A repeated row doubles that protein's weight, and set arithmetic hides it."""
    with pytest.raises(CoregistrationError, match="Duplicate protids"):
        shared_index({"a": ["P1", "P1", "P2"], "b": ["P1", "P2"]})


def test_no_spaces_at_all_is_an_error():
    with pytest.raises(CoregistrationError, match="at least one space"):
        shared_index({})


def test_an_unknown_reference_is_an_error_that_lists_the_options():
    with pytest.raises(CoregistrationError, match="nonesuch"):
        shared_index({"a": ["P1"], "b": ["P1"]}, reference="nonesuch")


def test_one_space_co_registers_with_itself():
    """Degenerate but legal: a config may compare a single space against nothing."""
    report = shared_index({"only": ["P1", "P2"]})
    assert report.index.as_list == ["P1", "P2"]
    assert report.is_exact


# ---------------------------------------------------------------------------
# what the report is for: aligning to it must then always succeed
# ---------------------------------------------------------------------------


def test_every_space_can_be_aligned_to_the_shared_index():
    """The payoff. `ProteinIndex.align` refuses to invent rows; after
    intersection it never has to, for any of the spaces."""
    import numpy as np

    space_protids = {"a": ["P1", "P2", "P3"], "b": ["P3", "P2", "P9"]}
    report = shared_index(space_protids)
    for space_id, protids in space_protids.items():
        values = np.arange(len(protids) * 2, dtype=np.float64).reshape(len(protids), 2)
        aligned = report.index.align(protids, values, what=space_id)
        assert aligned.shape == (report.n_shared, 2)


def test_alignment_actually_reorders_rather_than_merely_fitting():
    """Same shape is not the same rows. The row for P2 must follow P2."""
    import numpy as np

    report = shared_index({"a": ["P1", "P2"], "b": ["P2", "P1"]})
    b_values = np.array([[20.0], [10.0]])  # P2 then P1
    aligned = report.index.align(["P2", "P1"], b_values, what="b")
    assert report.index.as_list == ["P1", "P2"]
    assert aligned.tolist() == [[10.0], [20.0]]


def test_the_report_round_trips_through_its_dict_form():
    """It is written to disk beside the comparison it qualifies."""
    report = shared_index({"a": ["P1", "P2", "P3"], "b": ["P1", "P2"]})
    data = report.to_dict()
    assert data["n_shared"] == 2
    assert data["is_exact"] is False
    assert data["protids"] == ["P1", "P2"]
    assert data["spaces"][0] == {
        "space_id": "a",
        "n_own": 3,
        "n_dropped": 1,
        "dropped": ["P3"],
    }


def test_the_shared_index_is_a_protein_index():
    """So it carries the same refusal-to-invent-rows behavior everywhere."""
    report = shared_index({"a": ["P1"], "b": ["P1"]})
    assert isinstance(report.index, ProteinIndex)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_pairwise_distances_match_the_definition():
    """The Gram identity must give what the difference broadcast would have given."""
    rng = np.random.RandomState(0)
    values = rng.normal(size=(12, 5))
    expected = np.sqrt(((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2))
    assert np.allclose(pairwise_distances(values), expected, atol=1e-9)


def test_pairwise_distances_are_exactly_symmetric_with_a_zero_diagonal():
    """Not decoration: an asymmetry of one ulp gives `a` and `b` different
    neighbor lists for a reason no reader could see."""
    rng = np.random.RandomState(1)
    distances = pairwise_distances(rng.normal(size=(30, 8)) * 1e4)
    assert np.array_equal(distances, distances.T)
    assert np.array_equal(np.diag(distances), np.zeros(30))


def test_pairwise_distances_never_go_negative_on_identical_rows():
    """The Gram identity can produce a small negative squared distance, and
    `sqrt` of it is NaN, which then survives all the way to a coordinate."""
    values = np.tile(np.array([[1e6, 2e6, 3e6]]), (5, 1))
    distances = pairwise_distances(values)
    assert not np.isnan(distances).any()
    assert np.all(distances >= 0.0)


def test_k_nearest_excludes_the_protein_itself():
    values = np.array([[0.0], [1.0], [2.0], [10.0]])
    neighbors = k_nearest(pairwise_distances(values), k=1)
    assert neighbors[:, 0].tolist() == [1, 0, 1, 2]


def test_k_nearest_orders_by_distance():
    values = np.array([[0.0], [1.0], [5.0], [20.0]])
    neighbors = k_nearest(pairwise_distances(values), k=3)
    assert neighbors[0].tolist() == [1, 2, 3]


def test_k_nearest_breaks_ties_by_index_position():
    """Reproducible, and arbitrary. Both halves matter and both are documented."""
    values = np.array([[0.0], [1.0], [-1.0], [1.0]])
    neighbors = k_nearest(pairwise_distances(values), k=1)
    assert neighbors[0, 0] == 1  # positions 1, 2 and 3 are all at distance 1


@pytest.mark.parametrize("k", [0, -1, 4])
def test_an_impossible_k_is_rejected(k):
    values = np.arange(4.0).reshape(4, 1)
    with pytest.raises(CoregistrationError, match="k must be between"):
        k_nearest(pairwise_distances(values), k=k)


# ---------------------------------------------------------------------------
# neighborhood Jaccard
# ---------------------------------------------------------------------------


def test_a_space_compared_with_itself_agrees_perfectly():
    rng = np.random.RandomState(2)
    distances = pairwise_distances(rng.normal(size=(20, 4)))
    scores, _ = neighborhood_jaccard(distances, distances, k=5)
    assert np.array_equal(scores, np.ones(20))


def test_disjoint_neighborhoods_score_zero():
    """Two proteins on a line, two orthogonal orderings."""
    a = np.array([[0.0], [1.0], [10.0], [11.0]])
    b = np.array([[0.0], [10.0], [1.0], [11.0]])
    scores, _ = neighborhood_jaccard(pairwise_distances(a), pairwise_distances(b), k=1)
    assert scores[0] == 0.0  # nearest is 1 in `a` and 2 in `b`


def test_the_jaccard_denominator_is_the_union_not_k():
    """One shared neighbor out of two is 1/3, not 1/2."""
    a = np.array([[0.0], [1.0], [2.0], [100.0], [101.0]])
    b = np.array([[0.0], [1.0], [100.0], [2.0], [101.0]])
    scores, _ = neighborhood_jaccard(pairwise_distances(a), pairwise_distances(b), k=2)
    assert scores[0] == pytest.approx(1 / 3)


def test_boundary_ties_are_counted_rather_than_hidden():
    """Points on a symmetric line. The three interior ones have two equally
    near neighbors at k=1, so which one is "the" neighbor is index order.

    That is not a defect to fix -- there is no better answer -- but a mean
    Jaccard computed over mostly-tied neighborhoods is measuring the protein
    order, and the caller has to be able to see that it might be.
    """
    values = np.array([[0.0], [1.0], [-1.0], [2.0], [-2.0]])
    distances = pairwise_distances(values)
    _, diagnostics = neighborhood_jaccard(distances, distances, k=1)
    assert diagnostics["boundary_ties_a"] == 3  # 0, 1 and -1; the two ends are unambiguous
    assert diagnostics["k"] == 1


def test_no_boundary_ties_are_reported_when_every_distance_is_distinct():
    """Guard the guard: a count that is always positive would say nothing."""
    values = np.array([[0.0], [1.0], [4.0], [9.0], [16.0]])
    distances = pairwise_distances(values)
    _, diagnostics = neighborhood_jaccard(distances, distances, k=2)
    assert diagnostics["boundary_ties_a"] == 0


def test_mismatched_protein_counts_are_an_error_naming_the_fix():
    a = pairwise_distances(np.arange(10.0).reshape(10, 1))
    b = pairwise_distances(np.arange(8.0).reshape(8, 1))
    with pytest.raises(CoregistrationError, match="shared index"):
        neighborhood_jaccard(a, b, k=2)


# ---------------------------------------------------------------------------
# rank correlation
# ---------------------------------------------------------------------------


def test_rank_correlation_of_a_space_with_itself_is_one():
    rng = np.random.RandomState(3)
    distances = pairwise_distances(rng.normal(size=(15, 6)))
    scores, diagnostics = rank_correlation(distances, distances)
    assert np.allclose(scores, 1.0)
    assert diagnostics["undefined"] == 0


def test_a_monotone_rescaling_leaves_the_rank_correlation_at_one():
    """Spearman is about order. Doubling every coordinate changes no order."""
    rng = np.random.RandomState(4)
    values = rng.normal(size=(15, 3))
    a = pairwise_distances(values)
    b = pairwise_distances(values * 7.0)
    scores, _ = rank_correlation(a, b)
    assert np.allclose(scores, 1.0)


def test_a_reversed_ordering_gives_minus_one():
    n = 8
    a = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
    b = a.max() - a
    np.fill_diagonal(b, 0.0)
    scores, _ = rank_correlation(a, b)
    # Row 0's profile in `b` is the exact reverse of its profile in `a`.
    assert scores[0] == pytest.approx(-1.0)


def test_ties_share_a_mean_rank_rather_than_being_ordered_by_position():
    """Censored TM-scores arrive as exact zeros in their thousands. Ranking
    those by position manufactures an ordering out of file order."""
    from coregistration import average_ranks

    ranks = average_ranks(np.array([5.0, 1.0, 1.0, 1.0, 9.0]))
    assert ranks.tolist() == [3.0, 1.0, 1.0, 1.0, 4.0]


def test_a_protein_equidistant_from_everything_is_undefined_not_zero():
    """There is no ordering to correlate, and a 0.0 would read as disagreement."""
    n = 5
    a = np.ones((n, n))
    np.fill_diagonal(a, 0.0)
    rng = np.random.RandomState(5)
    b = pairwise_distances(rng.normal(size=(n, 3)))
    scores, diagnostics = rank_correlation(a, b)
    assert np.all(np.isnan(scores))
    assert diagnostics["undefined"] == n


def test_the_rank_correlation_matches_scipy_including_under_heavy_ties():
    """Cross-checked against the reference implementation, not merely plausible.

    The censored case is the one worth having: 60% of a production TM-score
    matrix is exactly zero (ADR 0009), so tie handling is not an edge case here,
    it is the common case. Skipped in the bare environment by design -- this
    module depends on no scipy, and the check is opportunistic.
    """
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.RandomState(42)
    values = np.where(rng.rand(25, 25) < 0.6, 0.0, rng.rand(25, 25))
    a = pairwise_distances(values)
    b = pairwise_distances(np.roll(values, 3, axis=1))

    ours, _ = rank_correlation(a, b)
    keep = ~np.eye(len(a), dtype=bool)
    theirs = np.array(
        [scipy_stats.spearmanr(a[i][keep[i]], b[i][keep[i]]).correlation for i in range(len(a))]
    )
    assert np.nanmax(np.abs(ours - theirs)) < 1e-12


def test_too_few_proteins_to_correlate_is_an_error():
    a = pairwise_distances(np.arange(2.0).reshape(2, 1))
    with pytest.raises(CoregistrationError, match="at least 3"):
        rank_correlation(a, a)


# ---------------------------------------------------------------------------
# Procrustes
# ---------------------------------------------------------------------------


def test_an_embedding_matches_itself_exactly():
    rng = np.random.RandomState(6)
    embedding = rng.normal(size=(20, 2))
    assert procrustes_disparity(embedding, embedding) == pytest.approx(0.0, abs=1e-12)


def test_rotation_translation_and_scale_are_all_free():
    """Procrustes exists to ignore exactly these three."""
    rng = np.random.RandomState(7)
    embedding = rng.normal(size=(20, 2))
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = (embedding @ rotation) * 3.5 + np.array([100.0, -40.0])
    assert procrustes_disparity(embedding, transformed) == pytest.approx(0.0, abs=1e-10)


def test_a_mirrored_layout_is_treated_as_the_same_map():
    """UMAP has no canonical handedness; forbidding reflection would report two
    identical maps as maximally different."""
    rng = np.random.RandomState(8)
    embedding = rng.normal(size=(20, 2))
    mirrored = embedding * np.array([1.0, -1.0])
    assert procrustes_disparity(embedding, mirrored) == pytest.approx(0.0, abs=1e-10)


def test_unrelated_layouts_score_well_above_zero():
    rng = np.random.RandomState(9)
    disparity = procrustes_disparity(rng.normal(size=(40, 2)), rng.normal(size=(40, 2)))
    assert 0.2 < disparity <= 1.0


def test_the_disparity_stays_inside_its_range():
    rng = np.random.RandomState(10)
    for seed in range(5):
        a = np.random.RandomState(seed).normal(size=(25, 2))
        b = rng.normal(size=(25, 2))
        assert 0.0 <= procrustes_disparity(a, b) <= 1.0


def test_embeddings_of_different_shapes_are_an_error():
    with pytest.raises(CoregistrationError, match="same shape"):
        procrustes_disparity(np.zeros((10, 2)), np.zeros((10, 3)))


def test_a_collapsed_embedding_is_an_error_rather_than_a_disparity_of_one():
    """A degenerate reduction is a different fact from two maps disagreeing."""
    with pytest.raises(CoregistrationError, match="collapsed"):
        procrustes_disparity(np.ones((10, 2)), np.random.RandomState(11).normal(size=(10, 2)))


def test_the_disparity_matches_scipy_when_scipy_is_installed():
    """The definition is scipy's, so the number must be scipy's.

    Skipped in the bare test environment, which is the point of ADR 0006: this
    module has no scipy dependency and the check is opportunistic.
    """
    scipy_spatial = pytest.importorskip("scipy.spatial")
    rng = np.random.RandomState(12)
    a = rng.normal(size=(30, 2))
    b = rng.normal(size=(30, 2))
    _, _, expected = scipy_spatial.procrustes(a, b)
    assert procrustes_disparity(a, b) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# the pair comparison
# ---------------------------------------------------------------------------


def make_features(seed, n=20, d=4):
    return np.random.RandomState(seed).normal(size=(n, d))


def test_comparing_a_space_with_itself_is_perfect_agreement():
    features = make_features(13)
    protids = [f"P{i}" for i in range(len(features))]
    result = compare_pair("a", features, "b", features, protids, k=5)
    assert np.array_equal(result.jaccard, np.ones(len(features)))
    assert np.allclose(result.spearman, 1.0)
    assert result.disparity is None  # no embeddings given


def test_procrustes_is_none_rather_than_fabricated_when_embeddings_are_absent():
    """Two spaces need not share a reducer, so the number is not always available."""
    features = make_features(14)
    protids = [f"P{i}" for i in range(len(features))]
    result = compare_pair("a", features, "b", make_features(15), protids, k=3)
    assert result.disparity is None
    assert result.diagnostics["procrustes_compared"] is False


def test_embeddings_bring_the_disparity_in():
    features_a, features_b = make_features(16), make_features(17)
    protids = [f"P{i}" for i in range(len(features_a))]
    result = compare_pair(
        "a",
        features_a,
        "b",
        features_b,
        protids,
        k=3,
        embedding_a=features_a[:, :2],
        embedding_b=features_b[:, :2],
    )
    assert result.disparity is not None
    assert 0.0 <= result.disparity <= 1.0
    assert result.diagnostics["procrustes_compared"] is True


def test_the_summary_carries_the_caveats_that_qualify_it():
    """The numbers are computed on unnormalized features under a euclidean
    metric neither block declared. That has to travel with them."""
    features = make_features(18)
    protids = [f"P{i}" for i in range(len(features))]
    summary = compare_pair("a", features, "b", make_features(19), protids, k=3).summary()
    assert summary["n_proteins"] == len(features)
    assert 0.0 <= summary["jaccard_mean"] <= 1.0
    assert any("normalization" in caveat for caveat in summary["geometry_caveats"])
    assert any("metric" in caveat for caveat in summary["geometry_caveats"])


# --- cluster-assignment ARI: Phase 6's fourth metric -------------------------


def _pair(clusters_a=None, clusters_b=None, n=48):
    """Two spaces over one index, optionally with partitions."""
    from coregistration import compare_pair
    from fusion_cohort import NARROW_BLOCK, WIDE_BLOCK, fusion_cohort

    cohort = fusion_cohort(n=n)
    return compare_pair(
        WIDE_BLOCK,
        cohort.values(WIDE_BLOCK),
        NARROW_BLOCK,
        cohort.values(NARROW_BLOCK),
        cohort.protids,
        k=5,
        clusters_a=clusters_a,
        clusters_b=clusters_b,
    ), cohort


def test_no_partitions_gives_no_ari_rather_than_a_number():
    """The rule Procrustes already follows: absent, not fabricated."""
    comparison, _ = _pair()
    assert comparison.cluster_ari is None
    assert comparison.summary()["cluster_ari"] is None
    assert comparison.diagnostics["clusters_compared"] is False


def test_two_spaces_that_agree_on_the_partition_score_one():
    from fusion_cohort import fusion_cohort

    labels = list(fusion_cohort(n=48).partitions["fold"].labels)
    comparison, _ = _pair(clusters_a=labels, clusters_b=labels)
    assert comparison.cluster_ari == pytest.approx(1.0)
    assert comparison.diagnostics["clusters_compared"] is True


def test_two_spaces_with_crossed_partitions_score_about_zero():
    """`fold` and `chemistry` are built as an exact cross-product, so a space
    recovering one carries no information about the other. This is the number
    that makes the metric worth having: two spaces can each be internally
    well-clustered and agree on nothing."""
    from fusion_cohort import fusion_cohort

    cohort = fusion_cohort()
    comparison, _ = _pair(
        clusters_a=list(cohort.partitions["fold"].labels),
        clusters_b=list(cohort.partitions["chemistry"].labels),
        n=cohort.n_proteins,
    )
    # At the fixture's default N=240; at N=48 the same exact crossing scores
    # -0.054, which is finite-sample noise in the index rather than structure.
    # The crossing is exact at any N divisible by 12 -- what changes with N is
    # how tightly ARI concentrates on 0, so the tolerance has to name an N.
    assert abs(comparison.cluster_ari) < 0.02


def test_the_ari_does_not_depend_on_the_label_names():
    from fusion_cohort import fusion_cohort

    cohort = fusion_cohort(n=48)
    plain = list(cohort.partitions["fold"].labels)
    renamed = [f"LC{value:02d}" for value in plain]
    comparison, _ = _pair(clusters_a=plain, clusters_b=renamed)
    assert comparison.cluster_ari == pytest.approx(1.0)


def test_a_degenerate_partition_withholds_the_ari_rather_than_reporting_one():
    """The demo's `families vs fused_late`, which read ARI 1.000.

    Both spaces put all eleven proteins in one cluster, and the adjusted Rand
    index of two such partitions is 1.0 by convention. Beside a neighborhood
    Jaccard of 0.291 that reads as "these two agree perfectly" when neither
    found any structure at all. Found by running the demo, not by a unit test.
    """
    labels = ["LC0"] * 48
    comparison, _ = _pair(clusters_a=labels, clusters_b=labels)
    assert comparison.cluster_ari is None
    assert comparison.diagnostics["clusters_compared"] is False
    assert "convention" in comparison.diagnostics["cluster_ari_note"]


def test_one_degenerate_side_also_withholds_it():
    """0.0 against a real partition is the same kind of convention, and reads
    as "these two disagree completely" instead of "one of them has nothing"."""
    from fusion_cohort import fusion_cohort

    real = list(fusion_cohort(n=48).partitions["fold"].labels)
    comparison, _ = _pair(clusters_a=["LC0"] * 48, clusters_b=real)
    assert comparison.cluster_ari is None
    assert "1 cluster(s)" in comparison.diagnostics["cluster_ari_note"]


def test_two_real_partitions_record_their_cluster_counts():
    from fusion_cohort import fusion_cohort

    cohort = fusion_cohort(n=48)
    comparison, _ = _pair(
        clusters_a=list(cohort.partitions["fold"].labels),
        clusters_b=list(cohort.partitions["chemistry"].labels),
    )
    assert comparison.cluster_ari is not None
    assert comparison.diagnostics["cluster_ari_note"] == "4 against 3 clusters"


def test_k_is_clamped_to_the_cohort_and_the_clamp_is_recorded():
    """A small shared index must not fail the rule, and must not deflate the score.

    The shared index is an intersection, so it is routinely smaller than either
    space: the default k of 10 against a 6-protein overlap used to raise. It now
    clamps -- but the denominator divides by k, so a clamp that reached the
    slice and not the denominator would report two *identical* spaces as
    disagreeing. Hence both assertions.
    """
    rng = np.random.RandomState(0)
    points = rng.normal(size=(6, 4))
    distances = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))

    scores, diagnostics = neighborhood_jaccard(distances, distances, k=10)
    assert diagnostics["k"] == 5, "k must clamp to N-1"
    assert diagnostics["k_requested"] == 10
    assert diagnostics["k_was_clamped"] is True
    np.testing.assert_allclose(scores, 1.0), "a space compared with itself must score 1.0"


def test_an_unclamped_k_is_left_alone_and_marked_as_such():
    rng = np.random.RandomState(1)
    points = rng.normal(size=(20, 4))
    distances = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    _, diagnostics = neighborhood_jaccard(distances, distances, k=5)
    assert diagnostics["k"] == 5
    assert diagnostics["k_was_clamped"] is False
