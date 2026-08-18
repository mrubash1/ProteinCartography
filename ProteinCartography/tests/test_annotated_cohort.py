"""The fixture's own ground truth, checked before any statistic consumes it.

A fixture that claims to have planted a signal, and has not, produces a test
suite that fails for the wrong reason and gets "fixed" by weakening the
statistic. A fixture that claims a column is null, and has accidentally
correlated it with the clustering, produces one that passes for the wrong
reason -- which is worse, because nothing ever fails.

So the generator's claims are checked here, in the terms the generator states
them: the planted rates, the planted effect sizes, the null columns, the two
encodings, and each of the five pathologies by name. §0.1's lesson, applied one
layer earlier: *a fixture can be statistically faithful and test nothing*, and
the way to find out is to measure it rather than read it.
"""

import ast

import numpy as np
import pandas as pd
import pytest
from annotated_cohort import (
    ALL_MISSING_COLUMN,
    CONSTANT_COLUMN,
    SINGLETON_PFAM_TERM,
    UNIVERSAL_LINEAGE_TERM,
    annotated_cohort,
    cluster_name,
    write_annotated_cohort,
)

#: The planted rates are drawn, not assigned, so the realised rate is a
#: binomial around the stated one. At n=50 per cluster a tolerance below about
#: 0.12 would make this test flaky rather than strict.
RATE_TOLERANCE = 0.12


@pytest.fixture(scope="module")
def cohort():
    return annotated_cohort()


def lineage_terms(value):
    """Parse the Python-list repr the pipeline writes into `Lineage`."""
    if not isinstance(value, str):
        return set()
    return set(ast.literal_eval(value))


def pfam_terms(value):
    """Parse the semicolon-terminated run the pipeline writes into `Pfam`."""
    if not isinstance(value, str):
        return set()
    return {term for term in value.split(";") if term}


def rate_of(frame, column, term, parse):
    """Fraction of *annotated* rows in `frame` carrying `term`."""
    annotated = frame[frame[column].notna()]
    if annotated.empty:
        return float("nan")
    return float(np.mean([term in parse(value) for value in annotated[column]]))


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_cohort():
    left = annotated_cohort(seed=7)
    right = annotated_cohort(seed=7)
    pd.testing.assert_frame_equal(left.frame, right.frame)


def test_a_different_seed_gives_a_different_cohort():
    """Guards the guard: a generator that ignored its seed would pass the test
    above and would make every downstream result an accident of one draw."""
    assert not annotated_cohort(seed=0).frame.equals(annotated_cohort(seed=1).frame)


def test_the_ground_truth_is_reported_alongside_the_data(cohort):
    assert cohort.signals() == {
        ("Lineage", "Chordata", "LC0"),
        ("Pfam", "PF00022", "LC3"),
        ("Length", None, "LC1"),
        ("Length", None, "LC5"),
    }


# ---------------------------------------------------------------------------
# the planted signal is really there
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", [0, 1])
def test_a_planted_term_is_present_at_the_rate_it_claims(cohort, index):
    planted = cohort.planted_terms[index]
    parse = lineage_terms if planted.column == "Lineage" else pfam_terms
    inside = cohort.members(planted.cluster)
    outside = cohort.frame[cohort.frame[cohort.cluster_column] != planted.cluster]

    assert rate_of(inside, planted.column, planted.term, parse) == pytest.approx(
        planted.rate_inside, abs=RATE_TOLERANCE
    )
    assert rate_of(outside, planted.column, planted.term, parse) == pytest.approx(
        planted.rate_outside, abs=RATE_TOLERANCE
    )


@pytest.mark.parametrize("index", [0, 1])
def test_a_planted_shift_moves_the_column_by_the_effect_size_it_claims(cohort, index):
    shift = cohort.planted_shifts[index]
    values = cohort.frame[shift.column]
    inside = values[cohort.frame[cohort.cluster_column] == shift.cluster].dropna()
    outside = values[cohort.frame[cohort.cluster_column] != shift.cluster].dropna()

    observed = (inside.mean() - outside.mean()) / values.std()
    # The comparison is one cluster against the other seven, so shifting one
    # cluster by d moves the contrast by slightly less than d, and shifting a
    # second cluster the other way moves the baseline too. A loose tolerance
    # here is honest; the sign and the rough magnitude are the claim.
    assert np.sign(observed) == np.sign(shift.effect_size)
    assert abs(observed) == pytest.approx(abs(shift.effect_size), rel=0.5)


def test_a_planted_cluster_keeps_its_annotation(cohort):
    """Missingness must never be applied to a cluster carrying that column's
    signal, or the fixture would be testing statistical power by accident."""
    for planted in cohort.planted_terms + cohort.planted_shifts:
        inside = cohort.members(planted.cluster)
        assert inside[planted.column].notna().all(), planted


# ---------------------------------------------------------------------------
# the null is really null
# ---------------------------------------------------------------------------


def test_organism_is_independent_of_the_clustering(cohort):
    """The column that must produce no findings. If the generator correlated it
    with the clustering, every enrichment test downstream would pass while
    proving nothing."""
    counts = pd.crosstab(cohort.frame[cohort.cluster_column], cohort.frame["Organism"])
    overall = counts.sum(axis=0) / counts.values.sum()
    per_cluster = counts.div(counts.sum(axis=1), axis=0)
    assert float((per_cluster - overall).abs().max().max()) < 0.20


def test_an_unplanted_term_is_not_concentrated_in_any_cluster(cohort):
    """`PF00125` is drawn at one rate everywhere and must stay that way."""
    rates = [
        rate_of(cohort.members(cluster), "Pfam", "PF00125", pfam_terms)
        for cluster in cohort.clusters
    ]
    assert max(rates) - min(rates) < 0.30


# ---------------------------------------------------------------------------
# the encodings
# ---------------------------------------------------------------------------


def test_lineage_is_a_python_list_repr_and_pfam_is_a_terminated_run(cohort):
    """The two encodings in one table, which is what the real file does."""
    lineage = cohort.frame["Lineage"].dropna().iloc[0]
    assert lineage.startswith("[") and lineage.endswith("]")
    assert isinstance(ast.literal_eval(lineage), list)

    pfam = next(value for value in cohort.frame["Pfam"] if value)
    assert pfam.endswith(";")
    # The trailing separator produces an empty final field. A parser that
    # splits and does not filter counts "" as a family in every protein.
    assert pfam.split(";")[-1] == ""


def test_a_protein_can_carry_no_families_at_all(cohort):
    """An empty `Pfam` is the common case in real data, not an error."""
    assert (cohort.frame["Pfam"] == "").any()


# ---------------------------------------------------------------------------
# the pathologies, one test each
# ---------------------------------------------------------------------------


def test_every_pathology_is_described(cohort):
    assert set(cohort.pathologies) == {
        "universal_term",
        "singleton_term",
        "cluster_with_no_measurements",
        "constant_column",
        "nested_taxonomy",
    }


def test_the_universal_term_is_carried_by_every_annotated_protein(cohort):
    annotated = cohort.frame["Lineage"].dropna()
    assert len(annotated) > 0
    assert all(UNIVERSAL_LINEAGE_TERM in lineage_terms(value) for value in annotated)


def test_the_singleton_term_is_carried_by_exactly_one_protein(cohort):
    carriers = [
        protid
        for protid, value in zip(cohort.frame["protid"], cohort.frame["Pfam"])
        if SINGLETON_PFAM_TERM in pfam_terms(value)
    ]
    assert len(carriers) == 1
    row = cohort.frame[cohort.frame["protid"] == carriers[0]].iloc[0]
    # Placed inside the enriched cluster on purpose: that is where a test
    # blind to the count is most likely to call it significant.
    assert row[cohort.cluster_column] == cohort.planted_terms[1].cluster


def test_one_cluster_has_no_measurements_for_one_column(cohort):
    empty = cluster_name(len(cohort.clusters) - 1, len(cohort.clusters))
    assert cohort.members(empty)[ALL_MISSING_COLUMN].isna().all()
    others = cohort.frame[cohort.frame[cohort.cluster_column] != empty]
    assert others[ALL_MISSING_COLUMN].notna().all()


def test_the_constant_column_has_no_variance(cohort):
    assert cohort.frame[CONSTANT_COLUMN].nunique() == 1


def test_the_taxonomy_is_nested(cohort):
    """No protein is a chordate without being a eukaryote. Two terms in a
    strict subset relation are not independent hypotheses, and a fixture with a
    flat vocabulary would never show that."""
    for value in cohort.frame["Lineage"].dropna():
        terms = lineage_terms(value)
        if "Chordata" in terms:
            assert "Eukaryota" in terms


# ---------------------------------------------------------------------------
# shape and the guards on the generator
# ---------------------------------------------------------------------------


def test_the_cohort_is_large_enough_to_have_power(cohort):
    """A p-value on eleven proteins is not evidence of anything (§0.5 note 3).
    This is the number that makes the fixture worth having."""
    assert len(cohort.frame) == 400
    assert len(cohort.clusters) == 8
    assert cohort.frame[cohort.cluster_column].value_counts().min() == 50


@pytest.mark.parametrize(
    "n_clusters,expected",
    [(2, "LC0"), (8, "LC0"), (10, "LC0"), (11, "LC00"), (100, "LC00"), (101, "LC000")],
)
def test_cluster_labels_are_padded_the_way_the_pipeline_pads_them(n_clusters, expected):
    """`leiden_clustering.py` widens the label to the digit count of the largest
    index, so the form changes at ten clusters. A fixture that hard-coded either
    form would stop matching its own planted labels when it crossed over."""
    assert cluster_name(0, n_clusters) == expected


def test_the_planted_labels_follow_the_cluster_count():
    """The regression the padding rule exists for."""
    wide = annotated_cohort(n=600, n_clusters=12)
    for planted in wide.planted_terms + wide.planted_shifts:
        assert planted.cluster in wide.clusters


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"n_clusters": 1}, "two is the minimum"),
        ({"n": 8, "n_clusters": 4}, "too small"),
    ],
)
def test_a_cohort_too_small_to_test_anything_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        annotated_cohort(**kwargs)


def test_it_round_trips_through_a_tsv(tmp_path, cohort):
    """Entry-point tests read it from disk, so the encodings have to survive."""
    path = write_annotated_cohort(tmp_path / "features" / "uniprot_features.tsv", cohort)
    reloaded = pd.read_csv(path, sep="\t")
    assert list(reloaded.columns) == list(cohort.frame.columns)
    assert lineage_terms(reloaded["Lineage"].dropna().iloc[0])
    assert reloaded[ALL_MISSING_COLUMN].isna().sum() == 50
