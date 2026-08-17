#!/usr/bin/env python
"""Co-registration: one protein index, several independent geometries.

Co-registration is the default product of this work (ADR 0001). Fusion mixes
several kinds of evidence into one set of coordinates and is an explicitly
invoked analysis; co-registration leaves each kind of evidence in its own space
and puts the *same* proteins in all of them, so that where two spaces disagree
is visible rather than averaged away.

That guarantee has a precondition nothing enforced until now. The four blocks
draw their protein sets from three different files -- `tmscore` from the
similarity matrix, `threedi` from the descriptor table, `biophys` and `domains`
from the UniProt features table -- and those files are produced by different
rules at different points in the pipeline. Two spaces built over slightly
different sets still reduce cleanly, still plot, and still look co-registered.
Every per-protein comparison between them would then be quietly conditioned on
an overlap nobody chose. `docs/FOLLOWUPS.md` #30 recorded this against group 7;
this module is the answer to it.

**The index is the intersection, and the loss is reported rather than hidden.**
Refusing outright to co-register spaces whose sets differ would be wrong: a
provider legitimately drops a protein it has no data for -- a structure with no
UniProt record, a sequence Foldseek could not fold -- and that is a fact about
the cohort, not a failure. What must never happen is losing those proteins
without saying so. So the intersection is taken, in the reference space's order,
and every dropped protein is named in the report.

The one hard error is an empty intersection, which means the spaces have no
protein in common and there is nothing to co-register.

## What gets compared, and at which level

Two of the three metrics here read a space's **feature matrix**, and one reads
its **2-D embedding**. That is not an inconsistency; the two levels answer
different questions and only one of them is available for each metric.

- `neighborhood_jaccard` and `rank_correlation` are properties of a space's own
  geometry, which is the feature matrix. Comparing them there is comparing what
  the space actually measured, before a reducer threw most of it away.
- `procrustes_disparity` needs both spaces in the *same* number of dimensions,
  and feature matrices are not: `biophys` has four columns, `domains` has one
  per observed family, `tmscore` has one per protein. The only shared
  dimensionality is the 2-D embedding, so Procrustes compares the pictures. It
  is the weaker of the three for exactly that reason, and it is the only one
  that answers "could a reader have superimposed these two plots by eye".

**The distances are computed on the features as the reducer consumes them:
unnormalized, and euclidean.** `spec.normalization` is recorded on every block
and applied nowhere -- `reduce_space` feeds `block.features` straight into PCA
-- and `spec.metric` is likewise never consulted (FOLLOWUPS #29). Applying
either here and not there would make this module describe a geometry that no
map is drawn from, which is worse than describing the real one imperfectly. The
report says so in `geometry_caveats` rather than leaving it to be inferred.

Everything below is numpy. Spearman, Procrustes and neighbor search all have
one-line scipy or scikit-learn equivalents, and ADR 0006 requires that the
default configuration run with neither installed.
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from index import IndexAlignmentError, ProteinIndex

__all__ = [
    "CoregistrationError",
    "CoregistrationReport",
    "PairComparison",
    "average_ranks",
    "compare_pair",
    "k_nearest",
    "neighborhood_jaccard",
    "pairwise_distances",
    "procrustes_disparity",
    "rank_correlation",
    "shared_index",
]

#: Stated on every comparison, because both are true of the geometry it
#: measured and neither is visible in the numbers.
GEOMETRY_CAVEATS = (
    "distances are euclidean: `spec.metric` is recorded on a block and never "
    "consulted by anything that reduces it (FOLLOWUPS #29)",
    "features are unnormalized: `spec.normalization` is recorded on a block and "
    "applied nowhere, so this describes the geometry the map is actually drawn "
    "from rather than the one the spec declares",
)


class CoregistrationError(Exception):
    """Raised when a set of spaces cannot be co-registered at all."""


@dataclass(frozen=True)
class SpaceContribution:
    """What one space brought to the shared index, and what it lost joining it.

    `dropped` is the interesting field. It is the proteins this space measured
    that at least one other space did not, so they cannot appear in any
    cross-space comparison. An empty tuple is the healthy case and the one the
    demo produces; a long tuple means two stages of the pipeline disagree about
    the cohort, which is worth knowing before reading any of the maps.
    """

    space_id: str
    n_own: int
    dropped: tuple

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)

    @property
    def is_complete(self) -> bool:
        """True when this space contributed every protein it had."""
        return not self.dropped

    def to_dict(self) -> dict:
        return {
            "space_id": self.space_id,
            "n_own": self.n_own,
            "n_dropped": self.n_dropped,
            "dropped": list(self.dropped),
        }


@dataclass(frozen=True)
class CoregistrationReport:
    """The shared index, plus an account of how each space reached it."""

    index: ProteinIndex
    reference: str
    contributions: tuple

    @property
    def n_shared(self) -> int:
        return len(self.index)

    @property
    def is_exact(self) -> bool:
        """True when every space had exactly the shared set, so nothing was lost."""
        return all(c.is_complete for c in self.contributions)

    def contribution(self, space_id: str) -> SpaceContribution:
        for c in self.contributions:
            if c.space_id == space_id:
                return c
        raise KeyError(space_id)

    def describe(self) -> str:
        """A short human-readable account, for stderr and for the review log."""
        if self.is_exact:
            return (
                f"{len(self.contributions)} spaces co-registered over "
                f"{self.n_shared} proteins; every space had the full set."
            )
        lines = [
            f"{len(self.contributions)} spaces co-registered over {self.n_shared} "
            f"proteins (reference: {self.reference!r}).",
            "",
            "Some spaces measured proteins the others did not. Those proteins are "
            "absent from every cross-space comparison below, so the comparison is "
            "conditioned on the overlap rather than on the cohort:",
        ]
        for c in self.contributions:
            if c.is_complete:
                continue
            shown = list(c.dropped)[:5]
            suffix = f" (+{c.n_dropped - 5} more)" if c.n_dropped > 5 else ""
            lines.append(f"  {c.space_id}: {c.n_own} own, {c.n_dropped} dropped: {shown}{suffix}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "n_shared": self.n_shared,
            "is_exact": self.is_exact,
            "protids": self.index.as_list,
            "spaces": [c.to_dict() for c in self.contributions],
        }


def shared_index(
    space_protids: Mapping[str, Sequence[str]],
    reference: str | None = None,
) -> CoregistrationReport:
    """The protein index every named space can be aligned to.

    Args:
        space_protids: space id -> that space's protids, in its own order.
        reference: which space's ordering the shared index inherits. Defaults to
            the first key. Order has to come from somewhere nameable: set
            iteration order is not reproducible, and sorting would silently
            reorder every space away from the order its blocks were computed in.

    Returns:
        A :class:`CoregistrationReport` whose `index` is the intersection in the
        reference space's order.

    Raises:
        CoregistrationError: no spaces given, an unknown reference, or an empty
            intersection.
        IndexAlignmentError: a space's own protids contain a duplicate.
    """
    if not space_protids:
        raise CoregistrationError(
            "no spaces to co-register. `coregistration.compare` needs at least "
            "one space, and comparing spaces needs at least two."
        )

    space_ids = list(space_protids)
    if reference is None:
        reference = space_ids[0]
    elif reference not in space_protids:
        raise CoregistrationError(
            f"coregistration reference {reference!r} is not among the spaces being "
            f"co-registered: {space_ids}."
        )

    # Building a ProteinIndex per space is not ceremony: it is where a duplicate
    # protid is caught. A repeated row doubles that protein's weight in a
    # geometry, and an intersection computed over sets would have discarded the
    # evidence that it happened.
    indexes = {}
    for space_id in space_ids:
        try:
            indexes[space_id] = ProteinIndex.from_iterable(space_protids[space_id])
        except IndexAlignmentError as error:
            raise CoregistrationError(f"space {space_id!r}: {error}") from error

    shared = indexes[reference]
    for space_id in space_ids:
        if space_id == reference:
            continue
        try:
            shared = shared.intersection(indexes[space_id])
        except IndexAlignmentError as error:
            raise CoregistrationError(
                f"spaces {reference!r} and {space_id!r} share no proteins, so there "
                f"is nothing to co-register: {error}"
            ) from error

    kept = set(shared.protids)
    contributions = tuple(
        SpaceContribution(
            space_id=space_id,
            n_own=len(indexes[space_id]),
            dropped=tuple(p for p in indexes[space_id] if p not in kept),
        )
        for space_id in space_ids
    )
    return CoregistrationReport(index=shared, reference=reference, contributions=contributions)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def pairwise_distances(values) -> np.ndarray:
    """The full ``(N, N)`` euclidean distance matrix of a feature matrix.

    Computed by the Gram identity rather than by broadcasting the differences.
    Broadcasting needs ``N * N * D`` intermediate floats, which for a 2500-protein
    `tmscore` profile block is hundreds of gigabytes; the Gram form needs
    ``N * N``.

    The identity loses a little precision near zero and is not exactly symmetric
    in floating point, so the result is clipped at zero and averaged with its
    transpose. Both matter downstream: a negative squared distance would produce
    a NaN, and an asymmetry of one ulp would let ``a``'s neighbor list disagree
    with ``b``'s for no reason a reader could see.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise CoregistrationError(f"expected an (N, D) feature matrix, got shape {values.shape}")
    square_norms = np.einsum("ij,ij->i", values, values)
    squared = square_norms[:, None] + square_norms[None, :] - 2.0 * (values @ values.T)
    np.maximum(squared, 0.0, out=squared)
    np.fill_diagonal(squared, 0.0)
    distances = np.sqrt(squared)
    return 0.5 * (distances + distances.T)


def k_nearest(distances: np.ndarray, k: int) -> np.ndarray:
    """The ``(N, k)`` positions of each protein's k nearest neighbors, self excluded.

    Ties are broken by index position, via a stable sort. That makes the result
    reproducible, which it must be, but it does not make an arbitrary choice
    less arbitrary: when the k-th and (k+1)-th neighbors are equidistant, which
    one lands in the set is decided by the reference space's protein order.
    :func:`neighborhood_jaccard` counts how often that happened and reports it.
    """
    n = distances.shape[0]
    if not 1 <= k <= n - 1:
        raise CoregistrationError(
            f"k must be between 1 and N-1 (N={n}), got {k}. A protein cannot be "
            "its own neighbor, so there are only N-1 candidates."
        )
    masked = distances.copy()
    np.fill_diagonal(masked, np.inf)
    return np.argsort(masked, axis=1, kind="stable")[:, :k]


def _boundary_ties(distances: np.ndarray, k: int) -> int:
    """Rows whose k-th and (k+1)-th neighbors are exactly equidistant."""
    n = distances.shape[0]
    if k >= n - 1:
        return 0
    masked = distances.copy()
    np.fill_diagonal(masked, np.inf)
    ordered = np.sort(masked, axis=1, kind="stable")
    return int(np.count_nonzero(ordered[:, k - 1] == ordered[:, k]))


# ---------------------------------------------------------------------------
# the three metrics
# ---------------------------------------------------------------------------


def neighborhood_jaccard(distances_a: np.ndarray, distances_b: np.ndarray, k: int = 10):
    """Per-protein agreement between two spaces' k-nearest-neighbor sets.

    This is the principled replacement for `calculate_concordance.py`, which
    subtracts a fraction sequence identity from a TM-score. Those are both
    numbers between 0 and 1 and they are not on the same scale, so their
    difference has no unit and no meaningful zero. A Jaccard index over
    neighbor sets asks the question that was actually meant -- do these two
    kinds of evidence put the same proteins next to this one -- and asks it in
    a quantity that is comparable across proteins, across spaces and across
    runs.

    Returns:
        ``(scores, diagnostics)``. `scores` is one Jaccard index per protein, in
        index order; 1.0 means the two spaces chose identical neighbor sets and
        0.0 means they chose disjoint ones. `diagnostics` records the boundary
        ties described in :func:`k_nearest`.
    """
    if distances_a.shape != distances_b.shape:
        raise CoregistrationError(
            f"the two spaces have different numbers of proteins: {distances_a.shape[0]} "
            f"and {distances_b.shape[0]}. They must be aligned to the shared index first."
        )
    neighbors_a = k_nearest(distances_a, k)
    neighbors_b = k_nearest(distances_b, k)

    scores = np.empty(neighbors_a.shape[0], dtype=np.float64)
    for row in range(neighbors_a.shape[0]):
        shared = len(set(neighbors_a[row].tolist()) & set(neighbors_b[row].tolist()))
        # |A n B| / |A u B|, and |A u B| = 2k - |A n B| when both sets have size k.
        scores[row] = shared / (2 * k - shared)
    diagnostics = {
        "k": k,
        "boundary_ties_a": _boundary_ties(distances_a, k),
        "boundary_ties_b": _boundary_ties(distances_b, k),
    }
    return scores, diagnostics


def average_ranks(row: np.ndarray) -> np.ndarray:
    """Ranks of `row`, with tied values sharing their mean rank.

    Tie handling is the whole reason this is not `argsort(argsort(x))`. Censored
    TM-scores arrive as exact zeros in their thousands, so a distance profile
    routinely has long runs of identical values; ranking those by position would
    manufacture an ordering out of the order the file happened to be written in,
    and two spaces would then correlate on that artifact.
    """
    order = np.argsort(row, kind="stable")
    ordered = row[order]
    ranks = np.empty(row.shape[0], dtype=np.float64)
    start = 0
    while start < ordered.shape[0]:
        stop = start
        while stop + 1 < ordered.shape[0] and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop)
        start = stop + 1
    return ranks


def rank_correlation(distances_a: np.ndarray, distances_b: np.ndarray):
    """Per-protein Spearman correlation between the two spaces' distance profiles.

    Where :func:`neighborhood_jaccard` asks about the k closest proteins, this
    asks about all of them: does the whole ordering agree, not just its head.
    The two disagree in a way worth seeing -- a space can order distant
    proteins identically while shuffling the near ones, which is the case that
    matters biologically and the one Jaccard is sensitive to.

    Returns:
        ``(scores, diagnostics)``. A protein whose distances are all equal in
        either space has no ordering to correlate and scores NaN; `diagnostics`
        counts those rather than letting them vanish into a mean.
    """
    if distances_a.shape != distances_b.shape:
        raise CoregistrationError(
            f"the two spaces have different numbers of proteins: {distances_a.shape[0]} "
            f"and {distances_b.shape[0]}. They must be aligned to the shared index first."
        )
    n = distances_a.shape[0]
    if n < 3:
        raise CoregistrationError(
            f"a distance profile over {n} proteins has at most {n - 1} other proteins "
            "in it, which is too few to correlate. Co-registration metrics need at "
            "least 3 shared proteins."
        )

    keep = ~np.eye(n, dtype=bool)  # drop each protein's zero distance to itself
    scores = np.empty(n, dtype=np.float64)
    for row in range(n):
        a = average_ranks(distances_a[row][keep[row]])
        b = average_ranks(distances_b[row][keep[row]])
        a = a - a.mean()
        b = b - b.mean()
        denominator = np.sqrt(float(a @ a) * float(b @ b))
        scores[row] = float(a @ b) / denominator if denominator > 0 else np.nan
    return scores, {"undefined": int(np.count_nonzero(np.isnan(scores)))}


def procrustes_disparity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """How well two 2-D embeddings superimpose, in ``[0, 1]``. Lower is closer.

    Both layouts are centered and scaled to unit Frobenius norm, then the
    rotation that best aligns them is found from the SVD of their cross-product.
    The disparity is the residual sum of squares, which reduces to
    ``1 - (sum of singular values)**2``. This is the definition
    `scipy.spatial.procrustes` uses, so the number is comparable to the standard
    one without depending on scipy.

    **Reflections are allowed**, and that is deliberate rather than an oversight
    in the derivation. UMAP's and t-SNE's output has no canonical handedness --
    the same run with a different seed routinely returns a mirrored layout --
    so forbidding reflection would report two identical maps as maximally
    different.
    """
    a = np.asarray(embedding_a, dtype=np.float64)
    b = np.asarray(embedding_b, dtype=np.float64)
    if a.shape != b.shape:
        raise CoregistrationError(
            f"Procrustes needs two embeddings of the same shape, got {a.shape} and "
            f"{b.shape}. Two spaces reduced with different reducers are not "
            "comparable this way."
        )
    if a.ndim != 2 or a.shape[0] < 2:
        raise CoregistrationError(f"expected an (N, D) embedding with N >= 2, got {a.shape}")

    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        raise CoregistrationError(
            "an embedding collapsed to a single point, so it has no shape to "
            "superimpose. This is a degenerate reduction, not a disparity of 1."
        )
    a /= norm_a
    b /= norm_b

    singular_values = np.linalg.svd(a.T @ b, compute_uv=False)
    disparity = 1.0 - float(singular_values.sum()) ** 2
    # Floating point can push an exact match a hair below zero.
    return float(np.clip(disparity, 0.0, 1.0))


# ---------------------------------------------------------------------------
# one pair of spaces, compared
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairComparison:
    """Every cross-space metric for one ordered pair, plus what qualifies them."""

    space_a: str
    space_b: str
    protids: tuple
    jaccard: np.ndarray
    spearman: np.ndarray
    disparity: float | None
    diagnostics: dict

    def summary(self) -> dict:
        """The scalar view, for a report a human reads before the per-protein table."""
        return {
            "space_a": self.space_a,
            "space_b": self.space_b,
            "n_proteins": len(self.protids),
            "jaccard_mean": float(np.mean(self.jaccard)),
            "jaccard_median": float(np.median(self.jaccard)),
            "spearman_mean": float(np.nanmean(self.spearman))
            if not np.all(np.isnan(self.spearman))
            else None,
            "procrustes_disparity": self.disparity,
            "diagnostics": self.diagnostics,
            "geometry_caveats": list(GEOMETRY_CAVEATS),
        }


def compare_pair(
    space_a: str,
    features_a,
    space_b: str,
    features_b,
    protids: Sequence[str],
    *,
    k: int = 10,
    embedding_a=None,
    embedding_b=None,
) -> PairComparison:
    """Compare two spaces already aligned to the shared index.

    `features_a` and `features_b` must have the same number of rows, in the same
    protein order -- align them through :func:`shared_index` first. The
    embeddings are optional because Procrustes needs both spaces reduced by the
    same reducer, which a config is not obliged to arrange; when they are absent
    the disparity is `None` rather than a fabricated number.
    """
    distances_a = pairwise_distances(features_a)
    distances_b = pairwise_distances(features_b)
    jaccard, jaccard_diagnostics = neighborhood_jaccard(distances_a, distances_b, k=k)
    spearman, spearman_diagnostics = rank_correlation(distances_a, distances_b)

    disparity = None
    if embedding_a is not None and embedding_b is not None:
        disparity = procrustes_disparity(embedding_a, embedding_b)

    return PairComparison(
        space_a=space_a,
        space_b=space_b,
        protids=tuple(protids),
        jaccard=jaccard,
        spearman=spearman,
        disparity=disparity,
        diagnostics={
            **jaccard_diagnostics,
            "spearman_undefined": spearman_diagnostics["undefined"],
            "procrustes_compared": disparity is not None,
        },
    )
