#!/usr/bin/env python
"""A generated cohort with annotations, clusters, and a known right answer.

The demo cohort cannot test enrichment. Eleven proteins, four of them
byte-identical in sequence, and a `Pfam` column with two distinct values across
the whole set (REVIEW_LOG G7.2) -- a p-value computed on that is not evidence of
anything, and a test asserting one would be asserting arithmetic rather than
behaviour. So the fixture comes before the statistic.

What this generates is a cohort where **the answer is known by construction**:
named terms are planted into named clusters at a stated marginal rate, named
continuous columns are shifted in named clusters by a stated effect size, and
everything else is drawn from a distribution that does not depend on the cluster
and must come back null. A statistic is then checkable in both directions -- it
has to find the planted signal *and* stay quiet everywhere else. Only the second
half catches a test that passes because everything came back significant.

Three properties are copied from the real table rather than invented, because
each has its own way of breaking a parser:

* **Two multi-value encodings in one table.** `Lineage` is a Python list
  *repr* -- ``"['cellular organisms', 'Eukaryota', ...]"`` -- and `Pfam` is a
  semicolon-*terminated* run, ``"PF00022;"``, which splits into a trailing empty
  string. Neither is the other and both are in `uniprot_features.tsv` today.
* **Missing values are ordinary.** A protein with no UniProt record has no
  lineage and no length, and the fraction that do is not small.
* **A nested taxonomy is nested.** `Chordata` implies `Eukaryota`, so the two
  are not independent hypotheses and any correction across them is
  conservative. The fixture is where that becomes visible rather than a
  footnote.

Four pathologies are planted deliberately, because each occurs in real data and
each has an answer that is easy to get wrong:

* a **universal term**, carried by every protein that is annotated at all,
  which cannot be enriched anywhere and whose p-value must be exactly 1.0
  rather than 0.0;
* a **singleton term**, carried by one protein, whose smallest attainable
  p-value is bounded well away from zero -- the case a minimum-count filter
  exists for, placed inside the enriched cluster where a count-blind test is
  most likely to call it;
* a **cluster with no measurements at all** for one continuous column, which
  must be reported as untested rather than compared against an invented value
  (`plot_cluster_distributions.remove_nans` substitutes ``(0,)`` here, putting a
  synthetic zero into a real test -- FOLLOWUPS #34);
* a **constant continuous column**, where every rank is tied and the normal
  approximation's variance term is zero.

Reproducible from the seed alone; :func:`annotated_cohort` writes nothing and
holds no state.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "AnnotatedCohort",
    "PlantedShift",
    "PlantedTerm",
    "annotated_cohort",
    "cluster_name",
    "write_annotated_cohort",
]

DEFAULT_N = 400
DEFAULT_CLUSTERS = 8
DEFAULT_SEED = 0

#: Carried by every annotated protein. Enrichment is impossible by construction.
UNIVERSAL_LINEAGE_TERM = "cellular organisms"

#: Carried by exactly one protein.
SINGLETON_PFAM_TERM = "PF99999"

#: Every value is NaN inside the last cluster and measured everywhere else.
ALL_MISSING_COLUMN = "pdb_confidence"

#: Every value is the same, everywhere.
CONSTANT_COLUMN = "Annotation"


def cluster_name(index: int, n_clusters: int) -> str:
    """`LC` plus the index, padded exactly as `leiden_clustering.py` pads it.

    The pipeline widens the label to the digit count of the *largest* cluster
    index, so eight clusters are `LC0`..`LC7` and twelve are `LC00`..`LC11`.
    Hard-coding either form gives a fixture that silently stops matching its own
    planted labels when the cluster count crosses ten.
    """
    return f"LC{index:0{len(str(max(0, n_clusters - 1)))}d}"


@dataclass(frozen=True)
class PlantedTerm:
    """A categorical term deliberately over-represented in one cluster.

    Both rates are **marginal** -- the fraction of proteins in that group
    carrying the term, not a rate conditioned on some earlier draw.
    """

    column: str
    term: str
    cluster: str
    rate_inside: float
    rate_outside: float

    @property
    def is_signal(self) -> bool:
        return self.rate_inside > self.rate_outside


@dataclass(frozen=True)
class PlantedShift:
    """A continuous column deliberately shifted in one cluster."""

    column: str
    cluster: str
    #: Difference in means, in units of the column's own standard deviation.
    effect_size: float

    @property
    def is_signal(self) -> bool:
        return self.effect_size != 0.0


@dataclass(frozen=True)
class AnnotatedCohort:
    """A cohort table plus the ground truth it was generated from."""

    frame: pd.DataFrame
    cluster_column: str
    categorical_columns: tuple
    continuous_columns: tuple
    planted_terms: tuple
    planted_shifts: tuple
    pathologies: dict = field(default_factory=dict)

    @property
    def protids(self) -> list:
        return list(self.frame["protid"])

    @property
    def clusters(self) -> list:
        """Cluster labels in sorted order, the order results are reported in."""
        return sorted(self.frame[self.cluster_column].unique())

    def signals(self) -> set:
        """`(column, term_or_None, cluster)` triples that must come back significant."""
        found = {(p.column, p.term, p.cluster) for p in self.planted_terms if p.is_signal}
        found |= {(s.column, None, s.cluster) for s in self.planted_shifts if s.is_signal}
        return found

    def members(self, cluster: str) -> pd.DataFrame:
        return self.frame[self.frame[self.cluster_column] == cluster]


def _lineage_repr(terms) -> str:
    """The Python-list repr UniProt metadata actually writes into the TSV."""
    return repr(list(terms))


def _pfam_run(terms) -> str:
    """The semicolon-*terminated* run `Pfam` and `InterPro` actually use."""
    return "".join(f"{term};" for term in terms)


def annotated_cohort(
    *,
    n: int = DEFAULT_N,
    n_clusters: int = DEFAULT_CLUSTERS,
    seed: int = DEFAULT_SEED,
    missing_fraction: float = 0.12,
) -> AnnotatedCohort:
    """Build the cohort. Deterministic given `seed`.

    `n` proteins in `n_clusters` equal contiguous clusters, labelled the way the
    pipeline labels them.
    """
    if n_clusters < 2:
        raise ValueError("enrichment compares a cluster against the rest; two is the minimum")
    if n < n_clusters * 4:
        raise ValueError(f"n={n} is too small to give {n_clusters} clusters anything to test")

    rng = np.random.RandomState(seed)
    cluster_of = np.arange(n) * n_clusters // n
    labels = np.array([cluster_name(index, n_clusters) for index in cluster_of])
    protids = [f"P{i:05d}" for i in range(n)]

    chordate_cluster = cluster_name(0, n_clusters)
    family_cluster = cluster_name(3 % n_clusters, n_clusters)
    up_cluster = cluster_name(1, n_clusters)
    down_cluster = cluster_name(5 % n_clusters, n_clusters)
    empty_cluster = cluster_name(n_clusters - 1, n_clusters)

    planted_terms = (
        # A taxon enriched in one cluster.
        PlantedTerm("Lineage", "Chordata", chordate_cluster, rate_inside=0.90, rate_outside=0.15),
        # A family enriched in a *different* cluster, so a bug that keys on the
        # first cluster or the first column reads as a miss rather than as a
        # coincidence.
        PlantedTerm("Pfam", "PF00022", family_cluster, rate_inside=0.85, rate_outside=0.10),
    )
    planted_shifts = (
        PlantedShift("Length", up_cluster, effect_size=2.0),
        PlantedShift("Length", down_cluster, effect_size=-1.5),
    )
    # Read the rates back off the ground truth rather than repeating the
    # literals, so the generator and what it claims to have generated cannot
    # drift apart.
    chordata, pf00022 = planted_terms

    frame = pd.DataFrame({"protid": protids, "LeidenCluster": labels})

    # -- Lineage: nested, so Chordata implies Eukaryota ----------------------
    # Chordata is drawn first, at its stated marginal rate, and the ranks above
    # it follow. Drawing Eukaryota first and Chordata conditionally would make
    # the recorded `rate_inside` a conditional rate that the ground truth then
    # misstates.
    lineages = []
    for label in labels:
        terms = [UNIVERSAL_LINEAGE_TERM]
        inside = label == chordata.cluster
        if rng.rand() < (chordata.rate_inside if inside else chordata.rate_outside):
            terms.extend(["Eukaryota", "Metazoa", "Chordata"])
        elif rng.rand() < 0.6:
            terms.append("Eukaryota")
            if rng.rand() < 0.4:
                terms.append("Fungi")
        else:
            terms.append("Bacteria")
        lineages.append(_lineage_repr(terms))
    frame["Lineage"] = lineages

    # -- Pfam: a semicolon-terminated run, plus one singleton term -----------
    families = []
    for label in labels:
        terms = []
        inside = label == pf00022.cluster
        if rng.rand() < (pf00022.rate_inside if inside else pf00022.rate_outside):
            terms.append(pf00022.term)
        if rng.rand() < 0.30:
            terms.append("PF00125")
        if rng.rand() < 0.20:
            terms.append("PF00071")
        families.append(terms)
    singleton_at = int(np.flatnonzero(labels == family_cluster)[0])
    families[singleton_at] = families[singleton_at] + [SINGLETON_PFAM_TERM]
    frame["Pfam"] = [_pfam_run(terms) for terms in families]

    # -- Organism: single-valued, and independent of the clustering ----------
    organisms = ["Homo sapiens", "Mus musculus", "Danio rerio", "Arabidopsis thaliana"]
    frame["Organism"] = [organisms[i] for i in rng.randint(0, len(organisms), size=n)]

    # -- Continuous columns --------------------------------------------------
    length_scale = 40.0
    length = rng.normal(loc=380.0, scale=length_scale, size=n)
    for shift in planted_shifts:
        length[labels == shift.cluster] += shift.effect_size * length_scale
    frame["Length"] = np.round(length, 1)

    frame[ALL_MISSING_COLUMN] = np.round(rng.uniform(60.0, 98.0, size=n), 2)
    frame[CONSTANT_COLUMN] = 5.0

    # -- The pathologies -----------------------------------------------------
    # One cluster has no pLDDT at all: every structure in it failed to
    # download. That comparison has no data, which is not the same as a
    # comparison that found nothing.
    frame.loc[frame["LeidenCluster"] == empty_cluster, ALL_MISSING_COLUMN] = np.nan

    # Ordinary scattered missingness, in both kinds of column, because a
    # protein with no UniProt record has none of this. The cluster carrying a
    # planted signal for a column is never blanked in that column -- the
    # fixture tests detection, not statistical power.
    for column in ("Lineage", "Length"):
        protected = np.zeros(n, dtype=bool)
        for planted in planted_terms + planted_shifts:
            if planted.column == column:
                protected |= labels == planted.cluster
        frame.loc[(rng.rand(n) < missing_fraction) & ~protected, column] = np.nan

    pathologies = {
        "universal_term": (
            f"{UNIVERSAL_LINEAGE_TERM!r} is carried by every protein that has a "
            "lineage at all, so it is universal within the tested universe and "
            "cannot be enriched in any cluster"
        ),
        "singleton_term": (
            f"{SINGLETON_PFAM_TERM!r} is carried by exactly one protein, and that "
            f"protein is in {family_cluster}, which is enriched for something else"
        ),
        "cluster_with_no_measurements": (
            f"every {ALL_MISSING_COLUMN} in {empty_cluster} is NaN, so that "
            "comparison has no data rather than no effect"
        ),
        "constant_column": (
            f"{CONSTANT_COLUMN} holds the same value everywhere, so every rank is "
            "tied and the variance term of the normal approximation is zero"
        ),
        "nested_taxonomy": (
            "'Chordata' implies 'Eukaryota', so the two are not independent "
            "hypotheses and a correction across both is conservative"
        ),
    }

    return AnnotatedCohort(
        frame=frame,
        cluster_column="LeidenCluster",
        categorical_columns=("Lineage", "Pfam", "Organism"),
        continuous_columns=("Length", ALL_MISSING_COLUMN, CONSTANT_COLUMN),
        planted_terms=planted_terms,
        planted_shifts=planted_shifts,
        pathologies=pathologies,
    )


def write_annotated_cohort(path, cohort: AnnotatedCohort) -> Path:
    """Write the cohort as the pipeline writes its feature tables."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cohort.frame.to_csv(path, sep="\t", index=False)
    return path
