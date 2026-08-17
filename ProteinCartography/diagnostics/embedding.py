#!/usr/bin/env python
"""How much of the geometry survived the drop to two dimensions.

Every map this pipeline produces is a projection of something with hundreds or
thousands of dimensions onto a page, and the projection is lossy in two
independent ways. Trustworthiness and continuity are the standard pair of
measures for them (Venna & Kaski 2001; Venna et al., JMLR 11:451, 2010), and
the reason there are two is that the two failures are not the same failure and
a map can have either without the other:

**Trustworthiness** asks about the neighbors the map *invented*. A protein's k
nearest neighbors on the page, which are far apart in the full space, are pairs
the reader will believe are related and which the data does not support. This
is the failure that produces wrong biology, because it puts things together.

**Continuity** asks about the neighbors the map *lost*. Proteins genuinely
close in the full space that the layout put far apart. This failure does not
assert anything false; it hides something true.

Both are in ``[0, 1]``, both are 1 for a perfect map, and both are computed per
protein and then averaged -- the global value is *exactly* the mean of the
per-protein ones, which is what makes "this protein's position is not
trustworthy" a statement you can make about one point rather than a share of a
number about the whole map.

The distinction is worth keeping because the remedies differ. A map with high
continuity and low trustworthiness is over-compressed: increase the perplexity
or the neighbor count, or read only the local structure. A map with high
trustworthiness and low continuity has torn a real group apart, which usually
means the reducer found a split the data does not have.

Cross-checked against ``sklearn.manifold.trustworthiness`` in
``tests/test_diagnostics_embedding.py``, which agrees exactly -- to 0.0e+00,
not to a tolerance -- on the four planted cases of ``tests/embedding_cohort.py``
at every k tried. The check is behind an ``importorskip`` because scikit-learn
is optional (ADR 0006), so it is also run in an environment where it does not
skip; a gated test can be written, be correct, and never execute.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_K",
    "DISTORTED_THRESHOLD",
    "FAITHFUL_THRESHOLD",
    "EmbeddingFaithfulness",
    "continuity",
    "faithfulness",
    "trustworthiness",
]

#: Matches ``coregistration``'s default so that "the 15 nearest neighbors"
#: means the same thing in the two places this work reports neighborhoods.
DEFAULT_K = 15

#: Reporting bands, not results. The literature offers no threshold for either
#: statistic, and inventing one and then quoting it as a standard would be
#: worse than saying where the line was drawn: at and above
#: :data:`FAITHFUL_THRESHOLD` the map is described as faithful in that
#: direction, at and below :data:`DISTORTED_THRESHOLD` as substantially
#: distorted, and in between as neither.
FAITHFUL_THRESHOLD = 0.90
DISTORTED_THRESHOLD = 0.70


class EmbeddingDiagnosticError(ValueError):
    """Raised when the two spaces cannot be compared at all."""


def _ordering(distances: np.ndarray) -> np.ndarray:
    """Each protein's neighbors, nearest first, with itself in column 0.

    Ranks and k-nearest sets are both derived from *this one* sort, which makes
    the identity they depend on structural rather than coincidental: a protein
    has rank <= k exactly when it is in the k-nearest set. Deriving them from
    two separate sorts would make that true only where there are no tied
    distances -- and censored TM-scores arrive as exact zeros in their
    thousands, so ties are the common case here rather than the corner.
    """
    masked = np.array(distances, dtype=np.float64, copy=True)
    np.fill_diagonal(masked, -np.inf)
    return np.argsort(masked, axis=1, kind="stable")


def _ranks_from(ordering: np.ndarray) -> np.ndarray:
    """``ranks[i, j]`` is j's position in i's ordering. Self is 0."""
    n = ordering.shape[0]
    ranks = np.empty((n, n), dtype=np.float64)
    ranks[np.arange(n)[:, None], ordering] = np.arange(n)[None, :]
    return ranks


def _check(high: np.ndarray, low: np.ndarray, k: int) -> int:
    if high.shape != low.shape or high.ndim != 2 or high.shape[0] != high.shape[1]:
        raise EmbeddingDiagnosticError(
            "both arguments must be the same square distance matrix shape, got "
            f"{high.shape} and {low.shape}. Align the two to a shared protein "
            "index before comparing them."
        )
    n = high.shape[0]
    # Above this the normalizing constant 2n - 3k - 1 is zero or negative and
    # the statistic is not defined. It is a real ceiling, not a safety margin:
    # at k = (2n-1)/3 the worst possible map and the best one score the same.
    ceiling = (2 * n - 1) / 3
    if not 1 <= k < ceiling:
        raise EmbeddingDiagnosticError(
            f"k must be at least 1 and less than (2N-1)/3 = {ceiling:.1f} for "
            f"N={n} proteins, got {k}. Above that the normalizing constant "
            "2N-3k-1 is not positive and neither statistic is defined."
        )
    return n


def _penalties(high: np.ndarray, low: np.ndarray, k: int) -> np.ndarray:
    """Per protein, its trustworthiness in ``[0, 1]``.

    The neighbors come from `low` and the ranks from `high`, which is the whole
    of the definition: how badly is each neighbor the *map* proposes ranked in
    the space the map claims to represent.
    """
    n = _check(high, low, k)
    ranks_high = _ranks_from(_ordering(high))
    neighbors_low = _ordering(low)[:, 1 : k + 1]

    ranks = np.take_along_axis(ranks_high, neighbors_low, axis=1)
    # rank <= k is exactly "also a k-nearest neighbor in the high-dimensional
    # space", so clipping at zero performs the set difference in the definition
    # without materializing the sets. Exact, not an approximation -- both sides
    # come from the same ordering.
    excess = np.clip(ranks - k, 0.0, None).sum(axis=1)

    # k * (2n - 3k - 1) / 2 is the largest excess one protein can accumulate,
    # so dividing by it puts every per-protein value on [0, 1] and makes the
    # mean of them equal the usual global statistic exactly.
    worst = k * (2.0 * n - 3.0 * k - 1.0) / 2.0
    return 1.0 - excess / worst


def trustworthiness(high: np.ndarray, low: np.ndarray, k: int = DEFAULT_K) -> np.ndarray:
    """Per protein: are the map's neighbors real? Returns ``(N,)`` in ``[0, 1]``.

    Args:
        high: pairwise distances in the space being represented.
        low: pairwise distances in the 2-D layout.
        k: neighborhood size.
    """
    return _penalties(np.asarray(high, dtype=np.float64), np.asarray(low, dtype=np.float64), k)


def continuity(high: np.ndarray, low: np.ndarray, k: int = DEFAULT_K) -> np.ndarray:
    """Per protein: did the map keep the real neighbors? ``(N,)`` in ``[0, 1]``.

    Exactly trustworthiness with the two spaces exchanged, which is not a
    shortcut but the definition: continuity penalizes proteins near in `high`
    and far in `low`, and trustworthiness penalizes the reverse. Implementing
    it this way means the two cannot drift apart, and it puts the whole weight
    of catching a swapped argument on the fixture -- ``tests/embedding_cohort``
    plants one case that fails each direction and passes the other, so an
    exchange here transposes a table the tests pin.
    """
    return _penalties(np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64), k)


@dataclass(frozen=True)
class EmbeddingFaithfulness:
    """What a 2-D layout kept and what it invented."""

    space_id: str
    reducer: str
    k: int
    protids: tuple
    trustworthiness: np.ndarray
    continuity: np.ndarray

    @property
    def n_proteins(self) -> int:
        return len(self.protids)

    @property
    def mean_trustworthiness(self) -> float:
        return float(self.trustworthiness.mean())

    @property
    def mean_continuity(self) -> float:
        return float(self.continuity.mean())

    def unreliable(self, threshold: float = DISTORTED_THRESHOLD) -> list:
        """Proteins whose position on the page should not be read, worst first."""
        worst = np.minimum(self.trustworthiness, self.continuity)
        order = np.argsort(worst, kind="stable")
        return [self.protids[i] for i in order if worst[i] <= threshold]

    def to_frame(self):
        """Per protein, as a tidy table. The file a disagreement mode reads."""
        import pandas as pd

        return pd.DataFrame(
            {
                "trustworthiness": self.trustworthiness,
                "continuity": self.continuity,
            },
            index=pd.Index(self.protids, name="protid"),
        )

    def to_dict(self) -> dict:
        return {
            "space_id": self.space_id,
            "reducer": self.reducer,
            "k": self.k,
            "n_proteins": self.n_proteins,
            "trustworthiness_mean": self.mean_trustworthiness,
            "trustworthiness_min": float(self.trustworthiness.min()),
            "continuity_mean": self.mean_continuity,
            "continuity_min": float(self.continuity.min()),
            "n_unreliable": len(self.unreliable()),
            "warnings": self.warnings(),
        }

    def warnings(self) -> list:
        """Plain language, because a number nobody reads is a number nobody acts on."""
        notes = []
        trust, keep = self.mean_trustworthiness, self.mean_continuity
        if trust <= DISTORTED_THRESHOLD:
            notes.append(
                f"trustworthiness is {trust:.2f}: the layout places proteins next to "
                "each other that are not close in the space it is drawing. Proximity "
                "on this map is not evidence of similarity."
            )
        if keep <= DISTORTED_THRESHOLD:
            notes.append(
                f"continuity is {keep:.2f}: the layout separates proteins that are "
                "close in the space it is drawing. Distance on this map is not "
                "evidence of dissimilarity, and a group may appear as two."
            )
        # Which of the two is worse says what kind of wrong the map is, and it
        # is the part a reader can act on. Only said when the gap is large
        # enough to mean something; a diagnostic that always fires is noise.
        gap = trust - keep
        if abs(gap) >= 0.10:
            if gap < 0:
                notes.append(
                    f"trustworthiness ({trust:.2f}) is well below continuity "
                    f"({keep:.2f}), so the layout is over-compressed rather than "
                    "torn: it is crowding distinct groups together, not splitting "
                    "one apart."
                )
            else:
                notes.append(
                    f"continuity ({keep:.2f}) is well below trustworthiness "
                    f"({trust:.2f}), so the layout is torn rather than crowded: what "
                    "it shows as two groups may be one. Treat any split in this map "
                    "as a hypothesis."
                )
        unreliable = self.unreliable()
        if unreliable:
            shown = ", ".join(unreliable[:5])
            suffix = f" (+{len(unreliable) - 5} more)" if len(unreliable) > 5 else ""
            notes.append(
                f"{len(unreliable)} of {self.n_proteins} proteins score at or below "
                f"{DISTORTED_THRESHOLD} on one of the two measures; their individual "
                f"positions should not be read: {shown}{suffix}"
            )
        if self.k >= self.n_proteins / 2:
            notes.append(
                f"k={self.k} is at least half of the {self.n_proteins} proteins, so "
                "these are not local neighborhoods and both numbers are closer to a "
                "statement about the whole cohort than about any protein in it."
            )
        if not notes:
            notes.append(
                f"the layout is faithful at k={self.k}: trustworthiness {trust:.2f}, "
                f"continuity {keep:.2f}."
            )
        return notes


def faithfulness(
    space_id: str,
    reducer: str,
    high: np.ndarray,
    low: np.ndarray,
    protids,
    k: int = DEFAULT_K,
) -> EmbeddingFaithfulness:
    """Both measures for one layout, per protein and in summary.

    Args:
        space_id: the space the layout belongs to.
        reducer: which reducer produced it, e.g. ``pca_umap``.
        high: pairwise distances in the space being represented.
        low: pairwise distances in the 2-D layout.
        protids: the shared protein order both matrices are in.
        k: neighborhood size.
    """
    protids = tuple(protids)
    if len(protids) != np.asarray(high).shape[0]:
        raise EmbeddingDiagnosticError(
            f"{len(protids)} protids but the distance matrix is "
            f"{np.asarray(high).shape[0]}x{np.asarray(high).shape[0]}."
        )
    return EmbeddingFaithfulness(
        space_id=space_id,
        reducer=reducer,
        k=k,
        protids=protids,
        trustworthiness=trustworthiness(high, low, k),
        continuity=continuity(high, low, k),
    )
