#!/usr/bin/env python
"""One high-dimensional truth, four 2-D maps of it with known faithfulness.

Trustworthiness and continuity are near-mirror formulas. Each walks the same
two neighbor sets and sums the same rank excess; they differ only in which
space supplies the neighbors and which supplies the ranks. The likeliest defect
in an implementation of them is therefore not an arithmetic slip -- it is
computing *one* of them twice under two names, or computing both correctly and
swapping the labels.

Neither defect is visible in a fixture where the map is simply good or simply
bad, because those give ``T == C`` and a wrong answer that agrees with itself
reads as two agreeing answers. Group 8a met the same shape of problem from the
other side: `graph` fusion returned a matrix that was symmetric, row-stochastic,
finite and correctly shaped, and carried no structure at all.
Only a planted partition it was *required* to recover could fail it.

So this fixture plants the 2x2 table, and every cell of it is occupied:

===========  =============  ============  ==========================================
case         trustworthy?   continuous?   what the map does to the truth
===========  =============  ============  ==========================================
`isometric`  yes            yes           nothing -- it is the truth, rotated
`fold`       **no**         yes           reflects, so distant regions touch
`split`      yes            **no**        tears a random half away, rigidly
`shuffle`    **no**         **no**        permutes the layout
===========  =============  ============  ==========================================

The off-diagonal cells are the load-bearing ones. `fold` invents neighbors
without separating any true ones; `split` separates true neighbors without
inventing any. An implementation that computes one statistic twice can only
produce the diagonal, and one that swaps the two produces the table transposed.

Measured at the defaults (N=240, seed 0), which is what
``tests/test_embedding_cohort.py`` pins:

============  ======  ======  ======  ======  ======  ======
case            T@5     C@5    T@10    C@10    T@20    C@20
============  ======  ======  ======  ======  ======  ======
`isometric`   1.0000  1.0000  1.0000  1.0000  1.0000  1.0000
`fold`        0.7970  0.9922  0.7948  0.9875  0.7967  0.9790
`split`       0.9909  0.6043  0.9847  0.6081  0.9724  0.6174
`shuffle`     0.4879  0.5133  0.4936  0.5155  0.5103  0.5169
============  ======  ======  ======  ======  ======  ======

Every cell clears :data:`FAITHFUL_THRESHOLD` by at least 0.072 in the direction
its answer demands, at every k above. The margin is stated rather than left
implicit because the four booleans are only a test if they are not marginal.

Two construction choices are load-bearing, and both were measured rather than
assumed:

**Generic position, no exact ties.** An earlier draft laid the truth on a 1-D
integer lattice, where every point is exactly equidistant from its two
neighbors. Tied distances make the k-nearest set depend on how the sort breaks
ties, so the `isometric` case scored 0.9993 instead of 1.0 and disagreed with
scikit-learn by up to 2.6e-3 -- not a formula error, purely a sorting artifact,
but indistinguishable from one. Drawn from a continuous distribution there are
no ties, `isometric` is exactly 1.0, and the agreement with scikit-learn is
exact to the last printed digit at every k tried.

**`split` translates a random half rigidly.** Every protein's displacement is
one of exactly two vectors, so within-half distances are preserved and every
neighbor the map proposes is one the truth also considers near -- that is what
holds trustworthiness at 0.98 while continuity collapses to 0.61. The
translation is bitwise exact; distances *recomputed* from the translated
coordinates are not, because ``(a + 500) - (b + 500)`` cancels inexactly for a
and b of order 1. The measured residual is 8.1e-13 relative, which is five
orders below anything a rank statistic can resolve. Splitting along a spatial
median instead was tried and does not work: it only separates points adjacent
to the cut, and continuity stays at 0.95.

Reproducible from the seed alone; :func:`embedding_cohort` writes nothing.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from spaces.base import BlockResult, BlockSpec

__all__ = [
    "FAITHFUL_THRESHOLD",
    "FOLD",
    "ISOMETRIC",
    "SHUFFLE",
    "SPLIT",
    "EmbeddingCase",
    "EmbeddingCohort",
    "embedding_cohort",
]

DEFAULT_SEED = 0

#: Matches ``tests/fusion_cohort.DEFAULT_N``. At N=120 the whole table still
#: holds, but `split`'s trustworthiness falls to 0.9355 at k=20 and the worst
#: margin halves to 0.036. Doubling N buys the margin back.
DEFAULT_N = 240

#: The truth is genuinely 2-D and is rotated into this many dimensions. The
#: rotation is orthonormal, so it changes no distance -- which is the point:
#: a diagnostic that reports the 2-D map of a 2-D structure as unfaithful is
#: reporting on its own arithmetic.
HIGH_DIMENSIONS = 20

ISOMETRIC = "isometric"
FOLD = "fold"
SPLIT = "split"
SHUFFLE = "shuffle"

#: Above this a map is called faithful in the named direction. Not a claim
#: about real data -- it is the threshold the planted 2x2 is separated by.
FAITHFUL_THRESHOLD = 0.90

#: How far `split` moves one half. Large enough that a displaced true neighbor
#: lands beyond every retained one, so continuity's penalty is the full rank
#: range rather than a quantity that depends on the cloud's diameter.
SPLIT_OFFSET = 500.0


@dataclass(frozen=True)
class EmbeddingCase:
    """One 2-D layout of the shared truth, and what it is required to score.

    ``trustworthy`` and ``continuous`` are the planted answer. They are what the
    diagnostic has to recover; the numbers themselves are incidental.
    """

    name: str
    low: np.ndarray
    trustworthy: bool
    continuous: bool
    description: str

    @property
    def faithful(self) -> bool:
        return self.trustworthy and self.continuous


@dataclass(frozen=True)
class EmbeddingCohort:
    """One truth in ``HIGH_DIMENSIONS`` dimensions, and four maps of it."""

    protids: list
    high: np.ndarray
    truth: np.ndarray
    cases: dict

    @property
    def n_proteins(self) -> int:
        return len(self.protids)

    def case(self, name: str) -> EmbeddingCase:
        return self.cases[name]

    def block_spec(self, block_id: str = "truth") -> BlockSpec:
        return BlockSpec(
            id=block_id,
            kind="features",
            fusable=True,
            metric="euclidean",
            normalization="none",
            provider="embedding_cohort",
        )

    def block_result(self, block_id: str = "truth") -> BlockResult:
        """The truth as a real :class:`BlockResult`, so tests drive the real path."""
        return BlockResult(
            spec=self.block_spec(block_id),
            protids=list(self.protids),
            features=np.asarray(self.high, dtype=np.float64),
        )

    def embedding_frame(self, name: str):
        """One case as the frame ``reduce_space`` writes.

        Same index name and same column naming as
        :meth:`spaces.reducers.core.ReducerResult.to_frame`, because a fixture
        that writes a shape the pipeline never produces is how group 7b passed
        36 unit tests and crashed on its first real run.
        """
        import pandas as pd

        low = self.cases[name].low
        return pd.DataFrame(
            low,
            index=pd.Index(self.protids, name="protid"),
            columns=[f"UMAP{i + 1}" for i in range(low.shape[1])],
        )

    def write_embedding(self, path, name: str) -> str:
        self.embedding_frame(name).to_csv(path, sep="\t")
        return str(path)


def _orthonormal_columns(rng: np.random.RandomState, rows: int, columns: int) -> np.ndarray:
    """An ``(rows, columns)`` matrix with orthonormal columns."""
    return np.linalg.qr(rng.normal(size=(rows, rows)))[0][:, :columns]


def embedding_cohort(*, n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> EmbeddingCohort:
    """Generate the cohort. Deterministic in ``seed`` alone."""
    if n < 12:
        raise ValueError(f"n must be at least 12 for the neighborhoods to mean anything, got {n}.")
    rng = np.random.RandomState(seed)
    protids = [f"EC{i:04d}" for i in range(n)]

    truth = rng.normal(size=(n, 2))
    high = truth @ _orthonormal_columns(rng, HIGH_DIMENSIONS, 2).T

    # Reflection about the vertical axis. Points at -x land on top of points at
    # +x, so the map claims two genuinely distant regions are adjacent. Nothing
    # is moved apart, so continuity is untouched.
    fold = truth.copy()
    fold[:, 0] = np.abs(fold[:, 0])

    # A random half, translated rigidly. Which half a protein lands in is
    # unrelated to where it sits, so roughly half of every protein's true
    # neighbors go with it and half do not.
    half = rng.randint(0, 2, size=n).astype(np.float64)
    split = truth + np.stack([half * SPLIT_OFFSET, np.zeros(n)], axis=1)

    shuffle = truth[rng.permutation(n)]

    cases = {
        ISOMETRIC: EmbeddingCase(
            ISOMETRIC,
            truth.copy(),
            trustworthy=True,
            continuous=True,
            description="the truth itself, before the rotation into high dimensions",
        ),
        FOLD: EmbeddingCase(
            FOLD,
            fold,
            trustworthy=False,
            continuous=True,
            description="reflected about x=0, so two distant regions are superimposed",
        ),
        SPLIT: EmbeddingCase(
            SPLIT,
            split,
            trustworthy=True,
            continuous=False,
            description="a random half translated away, rigidly and in full",
        ),
        SHUFFLE: EmbeddingCase(
            SHUFFLE,
            shuffle,
            trustworthy=False,
            continuous=False,
            description="the same points in a permuted layout, carrying no structure",
        ),
    }
    return EmbeddingCohort(protids=protids, high=high, truth=truth, cases=cases)
