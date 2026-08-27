"""Tests for the enrichment statistics.

Three layers, and each catches something the others cannot.

*Against scipy.* Every reimplemented statistic is cross-checked against the
reference implementation behind an `importorskip`, so the check runs wherever
scipy is installed and skips in the bare environment CI uses (ADR 0006). This
is the group 7 pattern; it caught nothing this time, which is the outcome that
makes it worth keeping.

*Against the definition.* The properties that hold regardless of implementation
-- a q-value never below its p-value, a universal term with p exactly 1.0, an
untested row staying out of the correction family -- because agreeing with
scipy on random inputs does not prove the edge cases were reached.

*Against the fixture.* The statistic run over `annotated_cohort`, which knows
where the signal is, checked in both directions: every planted signal found,
and nothing found in the columns that were generated null. The second half is
the one that fails when a test is passing for the wrong reason.
"""

import ast
import math

import numpy as np
import pytest
from annotated_cohort import (
    SINGLETON_PFAM_TERM,
    UNIVERSAL_LINEAGE_TERM,
    annotated_cohort,
)
from enrichment import (
    Comparison,
    benjamini_hochberg,
    detect_encoding,
    hypergeometric_enrichment,
    mann_whitney_u,
    normal_sf,
    parse_terms,
    term_counts,
)

SIGNIFICANT = 0.01


# ---------------------------------------------------------------------------
# parsing the two encodings that share a table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("PF00022;", ("PF00022",)),
        ("PF00022;PF00125;", ("PF00022", "PF00125")),
        # The trailing separator leaves an empty final field. A parser that
        # splits without filtering puts "" in every annotated protein.
        ("PF00022;;PF00125;", ("PF00022", "PF00125")),
        ("PF00022", ("PF00022",)),
        ("", ()),
        (";", ()),
        ("  PF00022 ; PF00125 ", ("PF00022", "PF00125")),
        # A term carried twice is carried by one protein.
        ("PF00022;PF00022;", ("PF00022",)),
    ],
)
def test_a_delimited_run_parses_to_its_terms(value, expected):
    assert parse_terms(value, "delimited") == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("['Eukaryota', 'Metazoa']", ("Eukaryota", "Metazoa")),
        ("[]", ()),
        ("['a']", ("a",)),
        # Order is the taxonomy's order and must survive.
        (
            "['cellular organisms', 'Eukaryota', 'Chordata']",
            ("cellular organisms", "Eukaryota", "Chordata"),
        ),
    ],
)
def test_a_list_repr_parses_to_its_terms(value, expected):
    assert parse_terms(value, "list_repr") == expected


def test_a_malformed_list_repr_becomes_one_opaque_term_rather_than_nothing():
    """Returning `()` would make an unparseable cell indistinguishable from an
    unannotated protein, which is the one thing the universe count must not get
    wrong."""
    assert parse_terms("['unclosed", "list_repr") == ("['unclosed",)


@pytest.mark.parametrize("encoding", ["single", "delimited", "list_repr"])
@pytest.mark.parametrize("missing", [None, float("nan"), "", "   ", "nan", "NaN"])
def test_a_missing_cell_carries_no_terms_under_any_encoding(encoding, missing):
    assert parse_terms(missing, encoding) == ()


def test_a_single_valued_cell_keeps_its_punctuation():
    """`Organism` is `Homo sapiens (Human)`. Splitting it would invent terms."""
    assert parse_terms("Homo sapiens (Human)", "single") == ("Homo sapiens (Human)",)


def test_an_unknown_encoding_is_refused():
    with pytest.raises(ValueError, match="unknown encoding"):
        parse_terms("x", "semicolon")


@pytest.mark.parametrize(
    "values,expected",
    [
        (["['a', 'b']", "['c']"], "list_repr"),
        (["PF1;", "PF2;PF3;"], "delimited"),
        (["Homo sapiens", "Mus musculus"], "single"),
        ([None, float("nan")], "single"),
        # One list among singles is not a list column; detection is per column.
        (["Homo sapiens", "['a']"], "single"),
    ],
)
def test_the_encoding_is_detected_from_the_whole_column(values, expected):
    assert detect_encoding(values) == expected


def test_detection_is_per_column_because_one_table_holds_both():
    cohort = annotated_cohort()
    assert detect_encoding(cohort.frame["Lineage"]) == "list_repr"
    assert detect_encoding(cohort.frame["Pfam"]) == "delimited"
    assert detect_encoding(cohort.frame["Organism"]) == "single"


def test_term_counts_counts_rows_not_occurrences():
    counts = term_counts(["A;B;", "A;A;", "", None], "delimited")
    assert counts == {"A": 2, "B": 1}


# ---------------------------------------------------------------------------
# Mann-Whitney, against scipy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["continuous", "heavily_tied", "censored"])
def test_mann_whitney_agrees_with_scipy(kind):
    """Including the two shapes real data actually has: heavy ties, and the
    60%-exact-zero censoring of ADR 0009. Tie handling is the common case here,
    not an edge case."""
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.RandomState(0)
    worst = 0.0
    for _ in range(100):
        n1, n2 = rng.randint(2, 60), rng.randint(2, 90)
        if kind == "continuous":
            a, b = rng.normal(0, 1, n1), rng.normal(0.4, 1, n2)
        elif kind == "heavily_tied":
            a = rng.randint(0, 4, n1).astype(float)
            b = rng.randint(0, 4, n2).astype(float)
        else:
            a = np.where(rng.rand(n1) < 0.6, 0.0, rng.rand(n1))
            b = np.where(rng.rand(n2) < 0.6, 0.0, rng.rand(n2))
        mine = mann_whitney_u(a, b)
        reference = scipy_stats.mannwhitneyu(a, b, method="asymptotic", alternative="two-sided")
        assert mine.statistic == pytest.approx(reference.statistic)
        worst = max(worst, abs(mine.p_value - reference.pvalue))
    assert worst < 1e-12


def test_mann_whitney_agrees_with_scipy_where_the_p_value_is_tiny():
    """Absolute agreement is not the claim that matters: a table is read at its
    top, where p is 1e-30 and an absolute tolerance of 1e-12 is vacuous."""
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.RandomState(1)
    checked = 0
    for _ in range(200):
        n1, n2 = rng.randint(20, 200), rng.randint(20, 300)
        a, b = rng.normal(0, 1, n1), rng.normal(1.2, 1, n2)
        reference = float(scipy_stats.mannwhitneyu(a, b, method="asymptotic").pvalue)
        if not 0 < reference < 1e-6:
            continue
        checked += 1
        assert mann_whitney_u(a, b).p_value == pytest.approx(reference, rel=1e-10)
    assert checked > 20, "the regime this test exists for was never reached"


def test_the_direction_of_the_effect_is_readable_from_its_sign():
    higher = mann_whitney_u([10, 11, 12, 13], [1, 2, 3, 4])
    lower = mann_whitney_u([1, 2, 3, 4], [10, 11, 12, 13])
    assert higher.effect == pytest.approx(1.0)
    assert lower.effect == pytest.approx(-1.0)
    assert higher.p_value == pytest.approx(lower.p_value)


def test_two_identical_distributions_have_no_effect():
    result = mann_whitney_u([1, 2, 3, 4], [1, 2, 3, 4])
    assert result.effect == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_ranks_are_tie_averaged_and_not_broken_by_position():
    """`argsort(argsort(x))` breaks ties by position, which turns a constant
    column into a strictly increasing rank vector and quietly deletes the tie
    correction. The two calls below differ only in the order of one group's
    values and must agree exactly."""
    left = mann_whitney_u([1.0, 1.0, 1.0, 2.0], [1.0, 1.0, 3.0])
    right = mann_whitney_u([2.0, 1.0, 1.0, 1.0], [3.0, 1.0, 1.0])
    assert left.p_value == pytest.approx(right.p_value)
    assert left.statistic == pytest.approx(right.statistic)


# ---------------------------------------------------------------------------
# Mann-Whitney, the branch that must not become a comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inside,outside,expected",
    [
        ([], [1, 2, 3], "no measurements in the cluster"),
        ([float("nan")] * 3, [1, 2, 3], "no measurements in the cluster"),
        ([1, 2, 3], [], "no measurements in the rest of the cohort"),
    ],
)
def test_a_group_with_no_measurements_is_untested_not_compared(inside, outside, expected):
    """FOLLOWUPS #34: `plot_cluster_distributions.remove_nans` substitutes
    `(0,)` here and runs the test anyway, so a cluster whose structures all
    failed to download reads as significantly low confidence rather than as
    unmeasured."""
    result = mann_whitney_u(inside, outside)
    assert not result.is_tested
    assert expected in result.note
    assert math.isnan(result.p_value)


def test_a_constant_column_is_untested_rather_than_a_division_by_zero():
    result = mann_whitney_u([5.0] * 10, [5.0] * 20)
    assert not result.is_tested
    assert "no ordering to test" in result.note


def test_missing_measurements_are_dropped_rather_than_ranked():
    with_nans = mann_whitney_u([1.0, np.nan, 2.0, 3.0], [10.0, 11.0, np.nan])
    without = mann_whitney_u([1.0, 2.0, 3.0], [10.0, 11.0])
    assert with_nans.p_value == pytest.approx(without.p_value)
    assert with_nans.n_inside == 3 and with_nans.n_outside == 2


# ---------------------------------------------------------------------------
# the hypergeometric tail
# ---------------------------------------------------------------------------


def test_hypergeometric_agrees_with_scipys_survival_function():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.RandomState(0)
    worst = 0.0
    for _ in range(300):
        universe = rng.randint(10, 800)
        carrying = rng.randint(0, universe + 1)
        n = rng.randint(1, universe + 1)
        k = rng.randint(max(0, n - (universe - carrying)), min(n, carrying) + 1)
        mine = hypergeometric_enrichment(k, n, carrying, universe)
        reference = float(scipy_stats.hypergeom.sf(k - 1, universe, carrying, n))
        worst = max(worst, abs(mine.p_value - reference))
    assert worst < 1e-10


def test_hypergeometric_agrees_with_scipy_where_the_p_value_is_tiny():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.RandomState(2)
    checked = 0
    for _ in range(600):
        universe = rng.randint(50, 2000)
        carrying = rng.randint(1, universe)
        n = rng.randint(2, min(universe, 400))
        k = rng.randint(max(0, n - (universe - carrying)), min(n, carrying) + 1)
        reference = float(scipy_stats.hypergeom.sf(k - 1, universe, carrying, n))
        if not 1e-300 < reference < 1e-6:
            continue
        checked += 1
        assert hypergeometric_enrichment(k, n, carrying, universe).p_value == pytest.approx(
            reference, rel=1e-9
        )
    assert checked > 50, "the regime this test exists for was never reached"


def test_it_is_the_same_number_as_a_one_sided_fisher_exact_test():
    """Named because a reader will recognise one of the two names."""
    scipy_stats = pytest.importorskip("scipy.stats")
    for k, n, carrying, universe in [(8, 10, 20, 100), (0, 10, 20, 100), (3, 17, 9, 120)]:
        table = [[k, n - k], [carrying - k, universe - carrying - (n - k)]]
        reference = scipy_stats.fisher_exact(table, alternative="greater")[1]
        assert hypergeometric_enrichment(k, n, carrying, universe).p_value == pytest.approx(
            reference
        )


def test_a_universal_term_cannot_be_enriched_anywhere():
    """Every protein carries it, so every cluster carries all of it, so the
    tail is the whole distribution. Exactly 1.0, not 0.9999999999 -- summing
    the terms would return the latter, which sorts to the top of a table
    ordered by p."""
    result = hypergeometric_enrichment(k=50, n=50, total_carrying=400, universe=400)
    assert result.p_value == 1.0
    assert result.effect == pytest.approx(1.0)


def test_a_term_absent_from_a_cluster_has_a_p_value_of_exactly_one():
    result = hypergeometric_enrichment(k=0, n=50, total_carrying=100, universe=400)
    assert result.p_value == 1.0
    assert result.effect == pytest.approx(0.0)


def test_a_singleton_term_cannot_reach_significance():
    """One protein carrying a term is the best possible evidence for that term
    and it is still weak. This is the bound a minimum-count filter exists to
    respect rather than to discover."""
    result = hypergeometric_enrichment(k=1, n=50, total_carrying=1, universe=400)
    assert result.p_value == pytest.approx(50 / 400)
    assert result.p_value > SIGNIFICANT


def test_fold_enrichment_reads_depletion_off_a_one_sided_table():
    """The one-sided p-value says nothing about under-representation; the
    effect does, which is why it is reported beside it."""
    depleted = hypergeometric_enrichment(k=1, n=50, total_carrying=200, universe=400)
    assert depleted.effect < 1.0
    assert depleted.p_value > 0.9


def test_a_column_nobody_is_annotated_for_is_untested():
    result = hypergeometric_enrichment(k=0, n=0, total_carrying=0, universe=0)
    assert not result.is_tested
    assert "annotated" in result.note


@pytest.mark.parametrize(
    "args", [(11, 10, 20, 100), (5, 10, 3, 100), (5, 200, 20, 100), (-1, 10, 20, 100)]
)
def test_an_impossible_contingency_table_is_an_error_not_a_number(args):
    """These arise from a join that lost rows, and the wrong response is a
    plausible p-value."""
    with pytest.raises(ValueError, match="impossible contingency table"):
        hypergeometric_enrichment(*args)


# ---------------------------------------------------------------------------
# Benjamini-Hochberg
# ---------------------------------------------------------------------------


def test_benjamini_hochberg_agrees_with_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.RandomState(0)
    for trial in range(100):
        m = rng.randint(1, 200)
        p_values = rng.rand(m)
        if trial % 4 == 0:
            p_values[rng.rand(m) < 0.3] = p_values[0]
        reference = scipy_stats.false_discovery_control(p_values, method="bh")
        assert benjamini_hochberg(p_values) == pytest.approx(reference)


def test_an_untested_hypothesis_does_not_inflate_the_family():
    """The whole reason `note` exists. Counting an untested row would deflate
    every q-value in the table, and nothing on the blank row would say so."""
    with_gaps = benjamini_hochberg([0.01, 0.02, np.nan, 0.03, np.nan])
    without = benjamini_hochberg([0.01, 0.02, 0.03])
    assert np.isnan(with_gaps[[2, 4]]).all()
    assert with_gaps[[0, 1, 3]] == pytest.approx(without)


def test_a_q_value_is_never_smaller_than_its_p_value():
    rng = np.random.RandomState(3)
    p_values = rng.rand(500)
    assert np.all(benjamini_hochberg(p_values) >= p_values - 1e-12)


def test_the_correction_is_monotone_in_p():
    rng = np.random.RandomState(4)
    p_values = np.sort(rng.rand(200))
    q_values = benjamini_hochberg(p_values)
    assert np.all(np.diff(q_values) >= -1e-12)


def test_equal_p_values_receive_equal_q_values():
    q_values = benjamini_hochberg([0.02, 0.01, 0.02, 0.5])
    assert q_values[0] == pytest.approx(q_values[2])


def test_a_family_of_one_is_uncorrected():
    assert benjamini_hochberg([0.031]) == pytest.approx([0.031])


def test_a_family_of_nothing_is_all_nan():
    assert np.isnan(benjamini_hochberg([np.nan, np.nan])).all()


def test_q_values_are_capped_at_one():
    assert np.all(benjamini_hochberg([0.9, 0.95, 0.99]) <= 1.0)


# ---------------------------------------------------------------------------
# the normal tail
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("z", [-8.0, -3.0, -1.0, 0.0, 1.0, 1.959963984540054, 3.0, 8.0, 40.0])
def test_the_normal_survival_function_agrees_with_scipy(z):
    scipy_stats = pytest.importorskip("scipy.stats")
    expected = float(scipy_stats.norm.sf(z))
    if expected == 0.0:
        assert normal_sf(z) == 0.0
    else:
        assert normal_sf(z) == pytest.approx(expected, rel=1e-12)


def test_the_untested_sentinel_carries_no_numbers():
    result = Comparison.untested("because", n_inside=3, n_outside=4)
    assert not result.is_tested
    assert math.isnan(result.p_value) and math.isnan(result.effect)
    assert result.effect_kind == "none"
    assert (result.n_inside, result.n_outside) == (3, 4)


# ---------------------------------------------------------------------------
# against the fixture, in both directions
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cohort():
    return annotated_cohort()


def categorical_result(cohort, column, term, cluster, encoding):
    """One (column, term, cluster) hypothesis, assembled by hand.

    Deliberately not routed through `enrich_clusters`: this has to show that
    the *statistic* finds the planted signal, independently of whether the
    entry point assembles the table correctly. Group 6's lesson, applied in
    advance -- test the callee's path and the caller's path, because a defect
    in one is invisible to a test of the other.
    """
    annotated = cohort.frame[cohort.frame[column].notna()]
    inside = annotated[annotated[cohort.cluster_column] == cluster]
    carries = annotated[column].map(lambda value: term in parse_terms(value, encoding))
    inside_carries = inside[column].map(lambda value: term in parse_terms(value, encoding))
    return hypergeometric_enrichment(
        k=int(inside_carries.sum()),
        n=len(inside),
        total_carrying=int(carries.sum()),
        universe=len(annotated),
    )


def continuous_result(cohort, column, cluster):
    values = cohort.frame[column]
    is_member = cohort.frame[cohort.cluster_column] == cluster
    return mann_whitney_u(values[is_member], values[~is_member])


@pytest.mark.parametrize("index", [0, 1])
def test_every_planted_term_is_found_where_it_was_planted(cohort, index):
    planted = cohort.planted_terms[index]
    encoding = detect_encoding(cohort.frame[planted.column])
    result = categorical_result(cohort, planted.column, planted.term, planted.cluster, encoding)
    assert result.is_tested
    assert result.p_value < 1e-6
    assert result.effect > 2.0


@pytest.mark.parametrize("index", [0, 1])
def test_a_planted_term_is_not_found_in_the_clusters_it_was_not_planted_in(cohort, index):
    """The half that catches a statistic which calls everything significant."""
    planted = cohort.planted_terms[index]
    encoding = detect_encoding(cohort.frame[planted.column])
    others = [cluster for cluster in cohort.clusters if cluster != planted.cluster]
    p_values = [
        categorical_result(cohort, planted.column, planted.term, cluster, encoding).p_value
        for cluster in others
    ]
    assert all(p > SIGNIFICANT for p in p_values), dict(zip(others, p_values))


@pytest.mark.parametrize("index", [0, 1])
def test_every_planted_shift_is_found_with_the_right_sign(cohort, index):
    shift = cohort.planted_shifts[index]
    result = continuous_result(cohort, shift.column, shift.cluster)
    assert result.is_tested
    assert result.p_value < 1e-4
    assert np.sign(result.effect) == np.sign(shift.effect_size)


def test_no_shift_is_found_in_the_clusters_that_were_not_shifted(cohort):
    shifted = {shift.cluster for shift in cohort.planted_shifts}
    p_values = {
        cluster: continuous_result(cohort, "Length", cluster).p_value
        for cluster in cohort.clusters
        if cluster not in shifted
    }
    # Six unshifted clusters, so one p-value below 0.05 by chance is expected.
    # Correcting is what the table does; here the claim is only that nothing is
    # strongly significant.
    assert all(p > SIGNIFICANT for p in p_values.values()), p_values


def test_the_organism_column_produces_no_findings(cohort):
    """Generated independently of the clustering, so every one of its 32
    hypotheses must survive correction."""
    terms = sorted(term_counts(cohort.frame["Organism"], "single"))
    p_values = [
        categorical_result(cohort, "Organism", term, cluster, "single").p_value
        for term in terms
        for cluster in cohort.clusters
    ]
    assert np.nanmin(benjamini_hochberg(p_values)) > 0.05


def test_the_universal_term_in_the_fixture_is_never_enriched(cohort):
    for cluster in cohort.clusters:
        result = categorical_result(cohort, "Lineage", UNIVERSAL_LINEAGE_TERM, cluster, "list_repr")
        assert result.p_value == 1.0


def test_the_singleton_term_in_the_fixture_survives_correction(cohort):
    """It sits inside the enriched cluster, which is where a count-blind test
    would be most tempted to call it."""
    cluster = cohort.planted_terms[1].cluster
    result = categorical_result(cohort, "Pfam", SINGLETON_PFAM_TERM, cluster, "delimited")
    assert result.p_value > SIGNIFICANT


def test_the_cluster_with_no_measurements_is_reported_untested(cohort):
    empty = cohort.clusters[-1]
    result = continuous_result(cohort, "pdb_confidence", empty)
    assert not result.is_tested
    assert "no measurements in the cluster" in result.note


def test_the_constant_column_is_untested_in_every_cluster(cohort):
    for cluster in cohort.clusters:
        assert not continuous_result(cohort, "Annotation", cluster).is_tested


def test_the_nested_taxon_is_also_enriched_and_that_is_not_a_bug(cohort):
    """`Chordata` implies `Eukaryota`, so planting the first raises the second
    in the same cluster. The two are not independent hypotheses and the
    correction across them is conservative -- worth a passing test rather than
    a footnote, because a reader who sees both rows will otherwise wonder which
    one is the finding."""
    chordata = categorical_result(cohort, "Lineage", "Chordata", "LC0", "list_repr")
    eukaryota = categorical_result(cohort, "Lineage", "Eukaryota", "LC0", "list_repr")
    assert chordata.p_value < eukaryota.p_value < SIGNIFICANT
    carriers = cohort.frame["Lineage"].dropna().map(lambda v: set(ast.literal_eval(v)))
    assert all("Eukaryota" in terms for terms in carriers if "Chordata" in terms)
