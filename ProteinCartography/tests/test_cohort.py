"""Tests for cohort selection.

The tests that matter here are the negative ones. Cohort selection sits upstream
of every block, so a rule that quietly reorders, deduplicates, or drops a
candidate changes the map with nothing downstream able to detect it -- which is
why several of these assert that the default rule does *not* tidy up.

The polarity tests are the other half. Ranking by e-value and ranking by
TM-score run in opposite directions, and getting it backwards produces a cohort
of the worst hits rather than the best, silently and plausibly.
"""

import dataclasses
import json

import pytest
from cohort import (
    LINEAGE_SHIFT_THRESHOLD,
    SIGNIFICANCE_MEASURES,
    CohortError,
    CohortReport,
    Selection,
    compare_lineage_composition,
    format_report,
    select,
)
from config_schema import SELECTION_RULES

# ---------------------------------------------------------------------------
# the default rule reproduces `accessions[:maximum]` exactly
# ---------------------------------------------------------------------------


def test_as_filtered_is_a_plain_prefix():
    candidates = ["Q9", "A1", "P0", "B2"]
    selection = select(candidates, max_structures=2)
    assert selection.retained == ("Q9", "A1")
    assert selection.discarded == ("P0", "B2")


def test_as_filtered_does_not_sort():
    """The whole reason the rule is named `as_filtered` and not `accession`."""
    selection = select(["Q9", "A1"], max_structures=1)
    assert selection.retained == ("Q9",)


def test_as_filtered_does_not_deduplicate():
    """Deduplicating would select a different set of proteins.

    Today a duplicate counts against `max_structures`. Removing it would admit
    one extra protein into the map, which looks like a tidy-up and is a change
    to the result.
    """
    selection = select(["A1", "A1", "B2"], max_structures=2)
    assert selection.retained == ("A1", "A1")
    assert selection.discarded == ("B2",)


def test_no_cap_retains_everything():
    selection = select(["B2", "A1"], max_structures=None)
    assert selection.retained == ("B2", "A1")
    assert selection.discarded == ()
    assert not selection.truncation_fired


def test_a_cap_larger_than_the_candidate_list_does_not_fire():
    selection = select(["A1", "B2"], max_structures=99)
    assert selection.discarded == ()
    assert not selection.truncation_fired


@pytest.mark.parametrize("rule", SELECTION_RULES)
def test_every_rule_partitions_the_candidates(rule):
    """Nothing is silently lost by any rule."""
    candidates = ["C3", "A1", "B2", "D4"]
    scores = {"A1": 1e-5, "B2": 1e-90, "C3": 1.0, "D4": 1e-40}
    selection = select(
        candidates,
        max_structures=2,
        rule=rule,
        scores=scores,
        measure="evalue" if rule == "significance" else None,
    )
    assert sorted(selection.retained + selection.discarded) == sorted(set(candidates))


# ---------------------------------------------------------------------------
# reproducibility is a property of the rule, and it is recorded
# ---------------------------------------------------------------------------


def test_as_filtered_truncation_is_flagged_as_not_reproducible():
    selection = select(["Q9", "A1", "P0"], max_structures=1)
    assert selection.truncation_fired
    assert not selection.reproducible


def test_as_filtered_without_truncation_is_reproducible():
    """When the cap decides nothing, the ordering decides nothing either."""
    selection = select(["Q9", "A1"], max_structures=10)
    assert not selection.truncation_fired
    assert selection.reproducible


def test_accession_truncation_is_reproducible():
    selection = select(["Q9", "A1", "P0"], max_structures=1, rule="accession")
    assert selection.retained == ("A1",)
    assert selection.reproducible


def test_accession_deduplicates():
    """Unlike `as_filtered`, this rule already changes membership by design."""
    selection = select(["A1", "A1", "B2"], max_structures=2, rule="accession")
    assert selection.retained == ("A1", "B2")


# ---------------------------------------------------------------------------
# significance, and the polarity that makes it dangerous
# ---------------------------------------------------------------------------


def test_evalue_ranks_lowest_first():
    scores = {"A1": 1e-3, "B2": 1e-90, "C3": 1.0}
    selection = select(
        ["A1", "B2", "C3"], max_structures=2, rule="significance", scores=scores, measure="evalue"
    )
    assert selection.retained == ("B2", "A1")


def test_tmscore_ranks_highest_first():
    """The same call shape as e-value, and the opposite direction."""
    scores = {"A1": 0.4, "B2": 0.95, "C3": 0.1}
    selection = select(
        ["A1", "B2", "C3"], max_structures=2, rule="significance", scores=scores, measure="tmscore"
    )
    assert selection.retained == ("B2", "A1")


def test_bits_ranks_highest_first():
    scores = {"A1": 100.0, "B2": 3000.0}
    selection = select(
        ["A1", "B2"], max_structures=1, rule="significance", scores=scores, measure="bits"
    )
    assert selection.retained == ("B2",)


def test_every_declared_measure_has_a_direction():
    for measure, spec in SIGNIFICANCE_MEASURES.items():
        assert spec["better"] in ("lower", "higher"), measure
        assert spec["description"]


def test_ties_are_broken_by_accession():
    """Without this the rule inherits the ordering it exists to escape."""
    scores = {"B2": 1e-10, "A1": 1e-10, "C3": 1e-10}
    first = select(
        ["B2", "A1", "C3"], max_structures=2, rule="significance", scores=scores, measure="evalue"
    )
    second = select(
        ["C3", "B2", "A1"], max_structures=2, rule="significance", scores=scores, measure="evalue"
    )
    assert first.retained == second.retained == ("A1", "B2")


def test_unscored_candidates_rank_last_and_are_counted():
    """Unscored is not the same as weak, so they are kept and reported."""
    scores = {"A1": 1e-10}
    selection = select(
        ["Z9", "A1", "B2"],
        max_structures=2,
        rule="significance",
        scores=scores,
        measure="evalue",
    )
    assert selection.retained == ("A1", "B2")
    assert selection.discarded == ("Z9",)
    assert selection.n_unscored == 2


def test_a_none_score_counts_as_unscored():
    scores = {"A1": 1e-10, "B2": None}
    selection = select(
        ["A1", "B2"], max_structures=2, rule="significance", scores=scores, measure="evalue"
    )
    assert selection.n_unscored == 1


def test_significance_without_scores_names_the_missing_input():
    with pytest.raises(CohortError, match="aggregate_hit_significance"):
        select(["A1"], max_structures=1, rule="significance", measure="evalue")


def test_significance_without_a_measure_lists_the_choices():
    with pytest.raises(CohortError, match="tmscore"):
        select(["A1"], max_structures=1, rule="significance", scores={"A1": 1.0})


def test_an_unknown_measure_is_rejected():
    with pytest.raises(CohortError, match="not one of"):
        select(
            ["A1"],
            max_structures=1,
            rule="significance",
            scores={"A1": 1.0},
            measure="pvalue",
        )


def test_an_unknown_rule_is_rejected():
    with pytest.raises(CohortError, match="as_filtered"):
        select(["A1"], max_structures=1, rule="best")


def test_a_nonpositive_cap_is_rejected():
    with pytest.raises(CohortError, match="must be positive"):
        select(["A1"], max_structures=0)


# ---------------------------------------------------------------------------
# lineage comparison
# ---------------------------------------------------------------------------


def lineages_for(retained_taxon, discarded_taxon, n=10):
    out = {}
    for i in range(n):
        out[f"R{i}"] = ["cellular organisms", "Eukaryota", retained_taxon]
        out[f"D{i}"] = ["cellular organisms", "Eukaryota", discarded_taxon]
    return out


def test_a_taxonomic_shift_is_detected():
    lineages = lineages_for("Metazoa", "Viridiplantae")
    comparison = compare_lineage_composition(
        [f"R{i}" for i in range(10)], [f"D{i}" for i in range(10)], lineages
    )
    assert comparison.max_abs_difference == pytest.approx(1.0)
    shifted = {term["term"] for term in comparison.shifted_terms()}
    assert shifted == {"Metazoa", "Viridiplantae"}


def test_a_term_shared_by_both_sets_is_not_a_shift():
    lineages = lineages_for("Metazoa", "Viridiplantae")
    comparison = compare_lineage_composition(
        [f"R{i}" for i in range(10)], [f"D{i}" for i in range(10)], lineages
    )
    shared = [t for t in comparison.terms if t["term"] == "Eukaryota"][0]
    assert shared["difference"] == 0.0


def test_an_identical_composition_reports_no_shift():
    lineages = lineages_for("Metazoa", "Metazoa")
    comparison = compare_lineage_composition(
        [f"R{i}" for i in range(10)], [f"D{i}" for i in range(10)], lineages
    )
    assert comparison.max_abs_difference == 0.0
    assert comparison.shifted_terms() == []


def test_a_rare_term_is_not_reportable():
    """One protein in a hundred should not be able to raise a taxonomic alarm."""
    lineages = {f"R{i}": ["Metazoa"] for i in range(100)}
    lineages["R0"] = ["Metazoa", "Rare"]
    lineages.update({f"D{i}": ["Metazoa"] for i in range(100)})
    comparison = compare_lineage_composition(
        [f"R{i}" for i in range(100)], [f"D{i}" for i in range(100)], lineages
    )
    rare = [t for t in comparison.terms if t["term"] == "Rare"][0]
    assert not rare["reportable"]
    assert comparison.max_abs_difference == 0.0


def test_proteins_without_a_lineage_are_excluded_from_both_denominators():
    """Missing lineage is missing data, not a taxon named "unknown"."""
    lineages = {"R0": ["Metazoa"], "R1": [], "D0": ["Viridiplantae"]}
    comparison = compare_lineage_composition(["R0", "R1"], ["D0"], lineages)
    assert comparison.n_retained_with_lineage == 1
    assert comparison.n_discarded_with_lineage == 1
    metazoa = [t for t in comparison.terms if t["term"] == "Metazoa"][0]
    assert metazoa["retained_proportion"] == 1.0


def test_no_lineage_data_at_all_reports_nothing_rather_than_zero():
    comparison = compare_lineage_composition(["R0"], ["D0"], {})
    assert comparison.terms == ()
    assert comparison.n_retained_with_lineage == 0


def test_the_term_table_is_ordered_by_effect_size_then_name():
    lineages = {
        "R0": ["Big", "Small", "Aaa"],
        "R1": ["Big", "Aaa"],
        "D0": ["Zzz"],
    }
    comparison = compare_lineage_composition(["R0", "R1"], ["D0"], lineages)
    differences = [abs(t["difference"]) for t in comparison.terms]
    assert differences == sorted(differences, reverse=True)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


def test_the_report_warns_that_the_default_cohort_is_not_reproducible():
    selection = select(["Q9", "A1", "P0"], max_structures=1)
    report = CohortReport.build(selection, ["Q9", "A1", "P0"])
    assert any("not controlled by this pipeline" in w for w in report.warnings())


def test_the_report_is_quiet_when_truncation_did_not_fire():
    selection = select(["Q9", "A1"], max_structures=10)
    report = CohortReport.build(selection, ["Q9", "A1"])
    assert report.warnings() == []


def test_the_report_warns_about_a_taxonomic_shift():
    candidates = [f"R{i}" for i in range(10)] + [f"D{i}" for i in range(10)]
    selection = select(candidates, max_structures=10)
    report = CohortReport.build(
        selection, candidates, lineages=lineages_for("Metazoa", "Viridiplantae")
    )
    assert any("taxonomically different" in w for w in report.warnings())


def test_the_report_warns_about_duplicates():
    selection = select(["A1", "A1", "B2"], max_structures=2)
    report = CohortReport.build(selection, ["A1", "A1", "B2"])
    assert any("duplicate" in w for w in report.warnings())


def test_the_report_warns_about_unscored_candidates():
    selection = select(
        ["A1", "B2", "C3"],
        max_structures=2,
        rule="significance",
        scores={"A1": 1e-9},
        measure="evalue",
    )
    report = CohortReport.build(selection, ["A1", "B2", "C3"])
    assert any("unscored, not weak" in w for w in report.warnings())


def test_the_report_records_the_three_counts():
    candidates = ["A1", "B2", "C3"]
    selection = select(candidates, max_structures=2)
    report = CohortReport.build(selection, candidates, n_candidates_before_filtering=99)
    payload = report.to_dict()
    assert payload["n_candidates_before_filtering"] == 99
    assert payload["n_candidates"] == 3
    assert payload["n_retained"] == 2
    assert payload["n_discarded"] == 1
    assert payload["truncation_fired"] is True
    assert payload["reproducible"] is False


def test_the_written_report_is_byte_stable(tmp_path):
    """The report lands in the output tree, so it must not perturb parity."""
    candidates = [f"R{i}" for i in range(10)] + [f"D{i}" for i in range(10)]
    selection = select(candidates, max_structures=10)
    report = CohortReport.build(
        selection, candidates, lineages=lineages_for("Metazoa", "Viridiplantae")
    )
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    report.write(str(first))
    report.write(str(second))
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text())["rule"] == "as_filtered"


def test_no_lineage_comparison_is_computed_when_nothing_was_discarded():
    """Comparing a full set against an empty one has no content."""
    candidates = ["R0", "R1"]
    selection = select(candidates, max_structures=10)
    report = CohortReport.build(selection, candidates, lineages=lineages_for("Metazoa", "X"))
    assert "lineage_comparison" not in report.to_dict()


def test_format_report_names_the_rule_and_the_counts():
    candidates = ["A1", "B2", "C3"]
    report = CohortReport.build(select(candidates, max_structures=2), candidates)
    text = format_report(report)
    assert "2 of 3 candidates retained" in text
    assert "rule=as_filtered" in text
    assert "NOT reproducible" in text


def test_the_shift_threshold_is_the_one_the_warning_uses():
    """Guards against the constant and the warning drifting apart."""
    lineages = {"R0": ["Metazoa"], "D0": ["Metazoa"]}
    comparison = compare_lineage_composition(["R0"], ["D0"], lineages)
    assert comparison.shifted_terms(LINEAGE_SHIFT_THRESHOLD) == []


def test_selection_is_frozen():
    selection = select(["A1"], max_structures=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        selection.rule = "accession"


def test_selection_can_be_built_directly_for_downstream_use():
    selection = Selection(retained=("A1",), discarded=(), rule="accession", max_structures=5)
    assert selection.reproducible
    assert not selection.truncation_fired
