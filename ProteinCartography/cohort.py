#!/usr/bin/env python
"""Which proteins reach the map, and what was discarded to get there.

Every space, every block, and every diagnostic downstream is conditioned on a
truncation that nobody currently sees: ``download_pdbs`` keeps the first
``max_structures`` accessions of the filtered hit list and drops the rest,
without a log line. The reasoning is in ``docs/adr/0008-cohort-selection.md``.

Two things in here are load-bearing rather than housekeeping.

**The default changes nothing.** ``as_filtered`` is a plain prefix of the list
in the order it arrives, which is what the pipeline does today. It is *not*
deduplicated and *not* sorted, because both would select a different set of
proteins -- a different map -- while looking like tidying up. The rule is named
``as_filtered`` rather than ``accession`` for the same reason: the sort that
PR #106 added upstream does not survive the round trip through UniProt, so
calling this "accession order" would repeat that mistake. What the default does
instead is *report*: how many candidates there were, whether truncation fired,
that the retained set is not reproducible, and how the discarded set differs
taxonomically from the retained one.

**Significance has a declared polarity.** A lower e-value is better and a higher
TM-score is better, so a bare ``scores`` mapping is a sign error waiting to
happen -- silent, and it inverts the cohort. :data:`SIGNIFICANCE_MEASURES` names
each measure and states its direction, the same way ``spaces.base`` fixes the
polarity of its per-cell channels, and :func:`select` derives the sort from that
table rather than from an argument the caller has to get right.

This module is deliberately free of pandas and of any file format: it takes
lists and mappings, and the caller reads the TSVs. That keeps it importable in
the minimal environment ``download_pdbs`` runs in, and testable without fixtures.
"""

from __future__ import annotations
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from config_schema import SELECTION_RULES, SIGNIFICANCE_MEASURES

__all__ = [
    "SIGNIFICANCE_MEASURES",
    "CohortError",
    "CohortReport",
    "LineageComparison",
    "Selection",
    "compare_lineage_composition",
    "format_report",
    "select",
]


class CohortError(Exception):
    """Cohort selection was asked for something it cannot do."""


#: Rules whose output does not depend on anything outside this repository.
#:
#: ``as_filtered`` is absent on purpose: its order is UniProt's response order,
#: which this repository does not control, so two runs of the same input can
#: retain different proteins.
REPRODUCIBLE_RULES = ("accession", "significance")

#: A lineage term must appear in at least this fraction of the retained or the
#: discarded set before a difference in the two proportions is worth reporting.
#: Below it, one or two proteins swing the proportion and the "shift" is noise.
LINEAGE_MIN_PREVALENCE = 0.05

#: How far apart the two proportions must be before the report calls it out.
#: A reporting threshold, not a test: nothing fails because of it, and the full
#: table is written to the report either way.
LINEAGE_SHIFT_THRESHOLD = 0.25


@dataclass(frozen=True)
class Selection:
    """The outcome of applying one selection rule to one candidate list."""

    retained: tuple
    discarded: tuple
    rule: str
    max_structures: int | None
    measure: str | None = None
    n_unscored: int = 0

    @property
    def truncation_fired(self) -> bool:
        return len(self.discarded) > 0

    @property
    def reproducible(self) -> bool:
        """Whether re-running the pipeline retains the same proteins.

        A rule that never truncates is trivially reproducible whatever its
        ordering, because the ordering decides nothing.
        """
        if not self.truncation_fired:
            return True
        return self.rule in REPRODUCIBLE_RULES


def select(
    candidates: Sequence[str],
    max_structures: int | None = None,
    rule: str = "as_filtered",
    scores: Mapping[str, float] | None = None,
    measure: str | None = None,
) -> Selection:
    """Choose which candidates reach the map.

    Args:
        candidates: accessions in the order the pipeline produced them. Order
            matters for ``as_filtered`` and is ignored by the other rules.
        max_structures: the cap, or None for no truncation.
        rule: one of :data:`config_schema.SELECTION_RULES`.
        scores: accession to raw score, required by ``significance``. Raw, not
            pre-inverted -- the direction comes from ``measure``.
        measure: which key of :data:`SIGNIFICANCE_MEASURES` ``scores`` holds.

    Returns:
        A :class:`Selection`. ``retained + discarded`` is a permutation of
        ``candidates`` for every rule, so nothing is silently lost.
    """
    if rule not in SELECTION_RULES:
        raise CohortError(f"selection rule {rule!r} is not one of {', '.join(SELECTION_RULES)}.")
    candidates = [str(candidate) for candidate in candidates]

    if max_structures is not None and max_structures <= 0:
        raise CohortError(f"max_structures must be positive, got {max_structures}.")

    if rule == "as_filtered":
        # Deliberately not deduplicated and not sorted. This is a byte-for-byte
        # reproduction of `accessions[:maximum]`, which is the only reason the
        # parity test can pass.
        ordered = list(candidates)
        n_unscored = 0
    elif rule == "accession":
        ordered = sorted(dict.fromkeys(candidates))
        n_unscored = 0
    else:
        ordered, n_unscored = _significance_order(candidates, scores, measure)

    if max_structures is None:
        retained, discarded = ordered, []
    else:
        retained, discarded = ordered[:max_structures], ordered[max_structures:]

    return Selection(
        retained=tuple(retained),
        discarded=tuple(discarded),
        rule=rule,
        max_structures=max_structures,
        measure=measure if rule == "significance" else None,
        n_unscored=n_unscored,
    )


def _significance_order(
    candidates: Sequence[str],
    scores: Mapping[str, float] | None,
    measure: str | None,
) -> tuple[list, int]:
    """Candidates best-first, with unscored ones last in accession order.

    Unscored candidates go last rather than being dropped. A candidate with no
    recorded evidence is one this pipeline failed to score, not one that scored
    badly, and dropping it would quietly conflate the two.
    """
    if measure is None:
        raise CohortError(
            "selection rule 'significance' needs a measure. "
            f"Choose one of {', '.join(sorted(SIGNIFICANCE_MEASURES))}."
        )
    if measure not in SIGNIFICANCE_MEASURES:
        raise CohortError(
            f"significance measure {measure!r} is not one of "
            f"{', '.join(sorted(SIGNIFICANCE_MEASURES))}."
        )
    if scores is None:
        raise CohortError(
            "selection rule 'significance' needs a table of scores, and none was "
            "given. In the pipeline that table is produced by "
            "`aggregate_hit_significance`; outside it, pass `scores=`."
        )

    better = SIGNIFICANCE_MEASURES[measure]["better"]
    sign = 1.0 if better == "lower" else -1.0

    unique = list(dict.fromkeys(candidates))
    scored = [c for c in unique if scores.get(c) is not None]
    has_score = set(scored)
    unscored = [c for c in unique if c not in has_score]

    # The accession tiebreak is what makes this reproducible. Without it, equal
    # scores would be ordered by whatever `sorted` was handed, which is the
    # UniProt-response order this rule exists to escape.
    scored.sort(key=lambda c: (sign * float(scores[c]), c))
    unscored.sort()
    return scored + unscored, len(unscored)


@dataclass(frozen=True)
class LineageComparison:
    """How the retained and discarded sets differ taxonomically.

    Reported per lineage *term* rather than per rank. Ranks are not comparable
    across kingdoms and picking a depth would impose a vocabulary; a term is
    just a string both sets either contain or do not, so the comparison needs no
    taxonomy of its own and degrades to "no data" rather than to a wrong answer.
    """

    n_retained_with_lineage: int
    n_discarded_with_lineage: int
    terms: tuple = ()

    @property
    def max_abs_difference(self) -> float:
        """The largest gap between the two proportions, over reportable terms.

        Named for its denominator rather than called "the" divergence. It is not
        a distance: proteins carry many terms at once, so the proportions do not
        sum to one and no total-variation-style summary is defined over them.
        """
        reportable = [t for t in self.terms if t["reportable"]]
        if not reportable:
            return 0.0
        return max(abs(t["difference"]) for t in reportable)

    def shifted_terms(self, threshold: float = LINEAGE_SHIFT_THRESHOLD) -> list:
        return [t for t in self.terms if t["reportable"] and abs(t["difference"]) >= threshold]

    def to_dict(self, top_n: int = 10) -> dict:
        return {
            "n_retained_with_lineage": self.n_retained_with_lineage,
            "n_discarded_with_lineage": self.n_discarded_with_lineage,
            "max_abs_proportion_difference": round(self.max_abs_difference, 6),
            "top_terms": [dict(t) for t in self.terms[:top_n]],
        }


def compare_lineage_composition(
    retained: Iterable[str],
    discarded: Iterable[str],
    lineages: Mapping[str, Sequence[str]],
) -> LineageComparison:
    """Compare the taxonomic makeup of the retained and discarded sets.

    Args:
        retained: accessions that reached the map.
        discarded: accessions truncation removed.
        lineages: accession to its lineage terms, most general first. A missing
            or empty entry means "no lineage recorded", and those proteins are
            excluded from both denominators rather than counted as a taxon.
    """
    retained, discarded = list(retained), list(discarded)
    retained_terms = _term_sets(retained, lineages)
    discarded_terms = _term_sets(discarded, lineages)
    n_retained, n_discarded = len(retained_terms), len(discarded_terms)

    if not n_retained or not n_discarded:
        return LineageComparison(n_retained, n_discarded, ())

    retained_counts = Counter(t for terms in retained_terms for t in terms)
    discarded_counts = Counter(t for terms in discarded_terms for t in terms)

    rows = []
    for term in set(retained_counts) | set(discarded_counts):
        retained_proportion = retained_counts.get(term, 0) / n_retained
        discarded_proportion = discarded_counts.get(term, 0) / n_discarded
        rows.append(
            {
                "term": term,
                "retained_proportion": round(retained_proportion, 6),
                "discarded_proportion": round(discarded_proportion, 6),
                "difference": round(retained_proportion - discarded_proportion, 6),
                "reportable": max(retained_proportion, discarded_proportion)
                >= LINEAGE_MIN_PREVALENCE,
            }
        )
    # Sorted by effect size, with the term name as a tiebreak so the report is
    # byte-stable across runs.
    rows.sort(key=lambda row: (-abs(row["difference"]), row["term"]))
    return LineageComparison(n_retained, n_discarded, tuple(rows))


def _term_sets(accessions: Sequence[str], lineages: Mapping[str, Sequence[str]]) -> list:
    """One deduplicated term set per accession that has a lineage at all."""
    out = []
    for accession in accessions:
        terms = lineages.get(accession)
        if not terms:
            continue
        cleaned = [str(term).strip() for term in terms if str(term).strip()]
        if cleaned:
            out.append(set(cleaned))
    return out


@dataclass(frozen=True)
class CohortReport:
    """Everything a reader needs to judge whether the cohort was fair."""

    selection: Selection
    n_candidates_before_filtering: int | None = None
    n_duplicate_candidates: int = 0
    lineage: LineageComparison | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        selection: Selection,
        candidates: Sequence[str],
        n_candidates_before_filtering: int | None = None,
        lineages: Mapping[str, Sequence[str]] | None = None,
        extra: Mapping | None = None,
    ) -> CohortReport:
        candidates = list(candidates)
        lineage = None
        if lineages and selection.truncation_fired:
            lineage = compare_lineage_composition(selection.retained, selection.discarded, lineages)
        return cls(
            selection=selection,
            n_candidates_before_filtering=n_candidates_before_filtering,
            n_duplicate_candidates=len(candidates) - len(set(candidates)),
            lineage=lineage,
            extra=dict(extra or {}),
        )

    def warnings(self) -> list:
        """Plain-language problems with this cohort, worst first."""
        selection = self.selection
        out = []
        if selection.truncation_fired and not selection.reproducible:
            out.append(
                f"{len(self.selection.discarded)} of "
                f"{len(selection.retained) + len(selection.discarded)} candidates were "
                f"discarded by the '{selection.rule}' rule, whose order is UniProt's "
                "response order. That order is not controlled by this pipeline, so "
                "re-running may produce a different map. Set `cohort.selection` to "
                "'accession' or 'significance' for a reproducible cohort."
            )
        if self.lineage is not None:
            shifted = self.lineage.shifted_terms()
            if shifted:
                worst = ", ".join(
                    f"{t['term']} ({t['retained_proportion']:.0%} retained vs "
                    f"{t['discarded_proportion']:.0%} discarded)"
                    for t in shifted[:3]
                )
                out.append(
                    "The discarded set is taxonomically different from the retained "
                    f"set: {worst}. Every clade claim from this run is conditioned on "
                    "that difference."
                )
        if selection.rule == "significance" and selection.n_unscored:
            out.append(
                f"{selection.n_unscored} candidate(s) had no {selection.measure} "
                "recorded and were ranked last. They are unscored, not weak."
            )
        if self.n_duplicate_candidates:
            out.append(
                f"the candidate list contains {self.n_duplicate_candidates} duplicate "
                "entries, which count against `max_structures` under the "
                "'as_filtered' rule exactly as they do today."
            )
        return out

    def to_dict(self) -> dict:
        selection = self.selection
        payload = {
            "rule": selection.rule,
            "max_structures": selection.max_structures,
            "measure": selection.measure,
            "n_candidates_before_filtering": self.n_candidates_before_filtering,
            "n_candidates": len(selection.retained) + len(selection.discarded),
            "n_retained": len(selection.retained),
            "n_discarded": len(selection.discarded),
            "n_duplicate_candidates": self.n_duplicate_candidates,
            "n_unscored": selection.n_unscored,
            "truncation_fired": selection.truncation_fired,
            "reproducible": selection.reproducible,
            "warnings": self.warnings(),
        }
        if self.lineage is not None:
            payload["lineage_comparison"] = self.lineage.to_dict()
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload

    def write(self, path: str) -> None:
        """Write the report as JSON, byte-stably."""
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")


def format_report(report: CohortReport) -> str:
    """The report as text, for a log a human actually reads."""
    payload = report.to_dict()
    lines = [
        "[cohort] {n_retained} of {n_candidates} candidates retained "
        "(rule={rule}, max_structures={max_structures})".format(**payload)
    ]
    if payload["truncation_fired"]:
        lines.append(
            "[cohort] truncation fired; retained set is "
            + ("reproducible" if payload["reproducible"] else "NOT reproducible")
        )
    for warning in payload["warnings"]:
        lines.append("[cohort] WARNING: " + warning)
    return "\n".join(lines)
