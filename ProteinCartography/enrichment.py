#!/usr/bin/env python
"""Cluster enrichment statistics, in numpy.

What a cluster is *made of* is a different question from where it sits, and the
pipeline currently answers it only as a picture: `plot_cluster_distributions.py`
runs a Mann-Whitney per cluster per numeric column and draws a star on a violin
plot. The numbers behind the stars are never written down, so nothing can sort
them, nothing can join them to anything, and the categorical annotations --
taxon, protein family -- are not tested at all even though they are the columns
a biologist actually asks about.

This module is the statistic that produces a table instead. It is deliberately
just the statistic: no I/O, no config, no cluster assignment. `enrich_clusters`
supplies those.

**Two tests, because there are two kinds of annotation.**

*Continuous* columns -- length, mean pLDDT -- get a Mann-Whitney U of one
cluster against every other protein, with the tie-corrected normal
approximation and a continuity correction. That is the generalization of what
the plotting script already does, and it is two-sided, because a cluster of
unusually short proteins is as much a finding as a cluster of long ones.

*Categorical* columns -- lineage, Pfam -- get the one-sided hypergeometric tail,
computed exactly. One-sided is a decision rather than an oversight: an
enrichment table answers "what is this cluster made of", and under-representation
of a term is dominated by terms that are rare everywhere, so a two-sided p-value
would fill the table with rows saying that a cluster of 50 does not contain a
family that only 3 proteins carry. Depletion is recoverable anyway -- every row
carries the observed and expected counts. See ADR 0012.

**No scipy.** ADR 0006: the framework has to import and run in an environment
with only numpy and pandas. Both tests are short in closed form, `math.lgamma`
and `math.erfc` are standard library, and `tests/test_enrichment.py` cross-checks
every one of them against scipy behind an `importorskip` so the check runs where
scipy exists and skips where it does not.

**A test that could not run is not a test that found nothing.** Every result
carries a `note`; when it is non-empty the test did not run and the reason says
why -- one side had no measurements, every value was identical, the column was
absent. This is the branch `plot_cluster_distributions.remove_nans` takes the
other way, substituting a synthetic ``0.0`` for a cluster with no data and then
testing it against real distributions (FOLLOWUPS #34). An untested row must
never be silently indistinguishable from a null result, and it must never enter
the multiple-testing family, because a hypothesis nobody tested is not a
hypothesis anyone can be wrong about.
"""

from __future__ import annotations
import ast
import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ENCODINGS",
    "Comparison",
    "benjamini_hochberg",
    "detect_encoding",
    "hypergeometric_enrichment",
    "mann_whitney_u",
    "normal_sf",
    "parse_terms",
    "term_counts",
]

#: How a cell holding more than one term is written. Both of the first two are
#: in `uniprot_features.tsv` today, in adjacent columns, which is why detection
#: is per column and overridable rather than guessed per value.
ENCODINGS = ("list_repr", "delimited", "single")


@dataclass(frozen=True)
class Comparison:
    """One group compared against another: one hypothesis, tested or explicitly not.

    `note` is empty when the test ran. When it is not, `p_value` is NaN and the
    row must be excluded from the correction family -- see `benjamini_hochberg`.

    Named for what it holds rather than the obvious `TestResult`, which pytest
    tries to collect as a test class and warns about on every run.
    """

    statistic: float
    p_value: float
    effect: float
    #: What `effect` measures, so a reader of the table never has to guess.
    effect_kind: str
    n_inside: int
    n_outside: int
    note: str = ""

    @property
    def is_tested(self) -> bool:
        return not self.note

    @classmethod
    def untested(cls, note: str, *, n_inside: int = 0, n_outside: int = 0) -> Comparison:
        return cls(
            statistic=math.nan,
            p_value=math.nan,
            effect=math.nan,
            effect_kind="none",
            n_inside=n_inside,
            n_outside=n_outside,
            note=note,
        )


# ---------------------------------------------------------------------------
# parsing a cell that holds more than one term
# ---------------------------------------------------------------------------


def parse_terms(value, encoding: str) -> tuple:
    """The set of terms in one cell, in the order they appear.

    Returns an empty tuple for anything missing, so "this protein has no
    annotation" and "this protein has an empty annotation" collapse -- they
    mean the same thing and distinguishing them would put an empty-string term
    in the vocabulary of every table.

    Order is preserved and duplicates are dropped: a term carried twice by one
    protein is carried by one protein.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}; expected one of {list(ENCODINGS)}")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ()

    if encoding == "single":
        found = [text]
    elif encoding == "list_repr":
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            # A malformed repr is data, not a crash. Treat the cell as one
            # opaque term so it stays visible in the table rather than
            # vanishing into an empty tuple that reads as "not annotated".
            found = [text]
        else:
            if isinstance(parsed, (list, tuple, set)):
                found = [str(item).strip() for item in parsed]
            else:
                found = [str(parsed).strip()]
    else:
        # Semicolon-*terminated*, so the last field is empty and must be
        # dropped. A parser that splits and does not filter puts "" in the
        # vocabulary of every protein that carries any family at all.
        found = [part.strip() for part in text.split(";")]

    seen = {}
    for term in found:
        if term:
            seen.setdefault(term, None)
    return tuple(seen)


def detect_encoding(values) -> str:
    """Guess how a column writes multiple terms, from the whole column.

    Per column, never per value. `Lineage` and `Pfam` sit next to each other in
    the same table with different encodings, and a per-value guess would read a
    single-valued organism name containing a semicolon as a list of two.
    """
    texts = [
        str(value).strip()
        for value in values
        if value is not None
        and not (isinstance(value, float) and math.isnan(value))
        and str(value).strip()
        and str(value).strip().lower() != "nan"
    ]
    if not texts:
        return "single"
    if all(text.startswith("[") and text.endswith("]") for text in texts):
        return "list_repr"
    if any(";" in text for text in texts):
        return "delimited"
    return "single"


def term_counts(values, encoding: str) -> dict:
    """`term -> number of rows carrying it`. Rows with no terms count nowhere."""
    counts = {}
    for value in values:
        for term in parse_terms(value, encoding):
            counts[term] = counts.get(term, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# the normal tail, without scipy
# ---------------------------------------------------------------------------


def normal_sf(z: float) -> float:
    """`P(Z > z)` for a standard normal. `erfc` is standard library and exact
    enough that the difference from scipy is at the last bit."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# continuous: Mann-Whitney U
# ---------------------------------------------------------------------------


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged.

    `argsort(argsort(x))` is the tempting one-liner and it is wrong in exactly
    the case that matters here: it breaks ties by position, which turns a column
    of equal values into a strictly increasing rank vector and destroys the tie
    correction the variance term depends on.
    """
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for stop in range(1, len(values) + 1):
        if stop == len(values) or sorted_values[stop] != sorted_values[start]:
            ranks[order[start:stop]] = 0.5 * (start + stop + 1)
            start = stop
    return ranks


def _tie_term(values: np.ndarray) -> float:
    """`sum(t**3 - t)` over tie groups, the correction the variance needs."""
    _, counts = np.unique(values, return_counts=True)
    counts = counts.astype(np.float64)
    return float(np.sum(counts**3 - counts))


def mann_whitney_u(inside, outside, *, use_continuity: bool = True) -> Comparison:
    """Two-sided Mann-Whitney U, tie-corrected normal approximation.

    Matches `scipy.stats.mannwhitneyu(..., method="asymptotic")`, including its
    choice to take the p-value from `max(U1, U2)` and to subtract the continuity
    correction from the numerator. The exact permutation p-value scipy's
    ``method="auto"`` prefers below n=8 is deliberately not reproduced: it is
    unreachable at the cohort sizes this runs on, and a statistic that switches
    definition partway up its own range is a bad thing to write into a table.

    `effect` is the rank-biserial correlation, `2*U1/(n1*n2) - 1`, which is the
    probability that a protein drawn from the cluster ranks above one drawn from
    the rest, rescaled to [-1, 1]. Positive means the cluster sits higher.

    NaNs are dropped -- they are missing measurements, not values. If dropping
    them empties either side the result is `untested`, never a comparison
    against a substituted zero (FOLLOWUPS #34).
    """
    inside = np.asarray(inside, dtype=np.float64)
    outside = np.asarray(outside, dtype=np.float64)
    inside = inside[~np.isnan(inside)]
    outside = outside[~np.isnan(outside)]

    n1, n2 = len(inside), len(outside)
    if n1 == 0 or n2 == 0:
        empty = "the cluster" if n1 == 0 else "the rest of the cohort"
        return Comparison.untested(
            f"no measurements in {empty}, so there is nothing to compare",
            n_inside=n1,
            n_outside=n2,
        )

    combined = np.concatenate([inside, outside])
    ranks = _average_ranks(combined)
    u1 = float(np.sum(ranks[:n1])) - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    effect = 2.0 * u1 / (n1 * n2) - 1.0

    n = n1 + n2
    tie_term = _tie_term(combined)
    variance = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1.0)))
    if variance <= 0.0:
        return Comparison.untested(
            "every value is identical, so there is no ordering to test",
            n_inside=n1,
            n_outside=n2,
        )

    numerator = max(u1, u2) - n1 * n2 / 2.0
    if use_continuity:
        numerator -= 0.5
    z = numerator / math.sqrt(variance)
    p_value = min(1.0, max(0.0, 2.0 * normal_sf(z)))

    return Comparison(
        statistic=u1,
        p_value=p_value,
        effect=effect,
        effect_kind="rank_biserial",
        n_inside=n1,
        n_outside=n2,
    )


# ---------------------------------------------------------------------------
# categorical: the exact hypergeometric tail
# ---------------------------------------------------------------------------


def _log_choose(n: float, k: float) -> float:
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


def hypergeometric_enrichment(k: int, n: int, total_carrying: int, universe: int) -> Comparison:
    """One-sided over-representation: `P(X >= k)` under the hypergeometric.

    Equal to `scipy.stats.hypergeom.sf(k-1, universe, total_carrying, n)` and to
    `fisher_exact(..., alternative="greater")`, computed by summing exact terms
    through `lgamma` rather than by any normal approximation -- the counts here
    are routinely small enough that an approximation is visibly wrong, and the
    sum is short.

    Args:
        k: proteins in this cluster carrying the term.
        n: proteins in this cluster (that are annotated for this column).
        total_carrying: proteins anywhere in the universe carrying the term.
        universe: annotated proteins in the cohort.

    `effect` is the fold enrichment, the observed rate over the cohort rate. It
    is 1.0 when the cluster looks exactly like the cohort, and it is what makes
    depletion readable off a one-sided table.
    """
    k, n, total_carrying, universe = int(k), int(n), int(total_carrying), int(universe)
    if universe <= 0 or n <= 0:
        return Comparison.untested(
            "no protein in this comparison is annotated for this column",
            n_inside=max(n, 0),
            n_outside=max(universe - n, 0),
        )
    if not 0 <= k <= min(n, total_carrying) or not n <= universe:
        raise ValueError(
            f"impossible contingency table: k={k}, n={n}, "
            f"total_carrying={total_carrying}, universe={universe}"
        )

    expected_rate = total_carrying / universe
    observed_rate = k / n
    effect = observed_rate / expected_rate if expected_rate > 0 else math.nan

    # P(X >= k). At k = 0 that is every outcome, so it is exactly 1 and the
    # sum is skipped -- floating point would otherwise return 0.9999999999.
    if k == 0:
        p_value = 1.0
    else:
        log_denominator = _log_choose(universe, n)
        upper = min(n, total_carrying)
        p_value = 0.0
        for i in range(k, upper + 1):
            p_value += math.exp(
                _log_choose(total_carrying, i)
                + _log_choose(universe - total_carrying, n - i)
                - log_denominator
            )
        p_value = min(1.0, max(0.0, p_value))

    return Comparison(
        statistic=float(k),
        p_value=p_value,
        effect=effect,
        effect_kind="fold_enrichment",
        n_inside=n,
        n_outside=universe - n,
    )


# ---------------------------------------------------------------------------
# multiple testing
# ---------------------------------------------------------------------------


def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, NaN-preserving.

    A NaN in means a NaN out, and -- this is the part that matters -- it does
    **not** count toward the family size. An untested hypothesis is not a
    hypothesis that came back null; counting it would deflate every other
    q-value in the table by inflating `m`, and the inflation would be invisible
    because the row it came from looks blank.

    Ties in the input get the same q-value, and the result is enforced
    monotone, so a larger p never receives a smaller q.
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(p_values.shape, np.nan, dtype=np.float64)

    testable = ~np.isnan(p_values)
    m = int(np.count_nonzero(testable))
    if m == 0:
        return adjusted

    values = p_values[testable]
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    # Walk down from the largest p, carrying the running minimum: q_(i) =
    # min over j >= i of p_(j) * m / j.
    scaled = ranked * m / np.arange(1, m + 1, dtype=np.float64)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    scaled = np.clip(scaled, 0.0, 1.0)

    # Equal p-values must receive equal q-values. The running minimum already
    # guarantees this for a stable sort, but only because ties are adjacent;
    # assign back through the same permutation to keep it that way.
    out = np.empty(m, dtype=np.float64)
    out[order] = scaled
    adjusted[testable] = out
    return adjusted
