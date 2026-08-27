#!/usr/bin/env python
"""Several blocks over one protein index, with a known right answer.

Fusion's failure mode is not that it crashes. It is that it produces a map
which looks fine and is one block's map wearing four blocks' labels (ADR 0002).
A fixture that only checks shapes cannot see that, and the demo cohort cannot
see it either -- eleven proteins whose `domains` block is nearly constant give
every strategy the same picture.

So this generates blocks where **what fusion should recover is fixed by
construction**, along two axes at once:

* **Two crossed partitions, one per block.** `fold` has four groups and is
  visible only in the `wide` block; `chemistry` has three and is visible only in
  `narrow`. They are crossed exactly -- every one of the twelve (fold,
  chemistry) cells holds the same number of proteins -- so neither partition
  carries any information about the other. A fusion that weights only `wide`
  must recover `fold` and stay blind to `chemistry`; weighting only `narrow`
  must do the reverse; weighting both must see both. Nothing about that is
  inferable from a single block, which is the point.
* **Scale and dimensionality deliberately incommensurate.** `wide` has 200
  columns at a scale of ~10 and `narrow` has 4 at ~0.01, so `wide`'s mean
  pairwise distance is roughly four orders of magnitude larger. The two blocks
  separate their own partitions about equally well, so any strategy that
  favours one of them is responding to units and column count rather than to
  information. This is precisely the case ADR 0002's normalization contract
  exists for, and the fixture makes it extreme enough that a broken
  normalization cannot hide inside sampling noise.

Four further blocks are here because each has an answer that is easy to get
wrong:

* `narrow_rescaled` is `narrow` multiplied by a constant. It carries *identical*
  information in different units, so `late` and `graph` must produce the same
  geometry from either -- exactly, not approximately. A normalization applied
  in the wrong place fails this and nothing else.
* `noise` carries no partition at all. Its contribution share must still be
  reported, and it must still sum with the others to 1: a block that
  contributes nothing to the structure is not thereby absent from the distance.
* `degenerate` has identical rows, so every pairwise distance is zero and the
  mean-distance normalization divides by zero. ADR 0002 promises a clear error
  here rather than a NaN geometry.
* `constant_column` is ordinary except that one of its columns never varies,
  which is where per-block standardization divides by zero.

Reproducible from the seed alone; :func:`fusion_cohort` writes nothing and holds
no state.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from spaces.base import BlockResult, BlockSpec

__all__ = [
    "CHEMISTRY_GROUPS",
    "CONSTANT_COLUMN_BLOCK",
    "DEGENERATE_BLOCK",
    "FOLD_GROUPS",
    "NARROW_BLOCK",
    "NOISE_BLOCK",
    "RESCALED_BLOCK",
    "RESCALE_FACTOR",
    "WIDE_BLOCK",
    "FusionCohort",
    "PlantedPartition",
    "fusion_cohort",
    "separation",
    "write_fusion_cohort",
]

DEFAULT_SEED = 0

#: 12 = 4 fold groups x 3 chemistry groups, so N must be a multiple of 12 for the
#: crossing to be exact. Everything below assumes it is; :func:`fusion_cohort`
#: enforces it rather than rounding, because an inexact crossing makes the two
#: partitions weakly dependent and that dependence would look like fusion
#: working.
DEFAULT_N = 240
FOLD_GROUPS = 4
CHEMISTRY_GROUPS = 3

WIDE_BLOCK = "wide"
NARROW_BLOCK = "narrow"
NOISE_BLOCK = "noise"
DEGENERATE_BLOCK = "degenerate"
RESCALED_BLOCK = "narrow_rescaled"
CONSTANT_COLUMN_BLOCK = "constant_column"

#: `narrow_rescaled` is this multiple of `narrow`. A power of ten so that the
#: rescaling is exact in binary floating point only up to the usual rounding --
#: which is the honest case, since a real unit change is not a power of two.
RESCALE_FACTOR = 1000.0

#: (columns, centroid scale, within-group noise scale) per structured block.
WIDE_SHAPE = (200, 10.0, 3.0)
NARROW_SHAPE = (4, 0.01, 0.003)


@dataclass(frozen=True)
class PlantedPartition:
    """A grouping of the proteins, and which block was built to show it.

    ``labels`` is one integer per protein, in the cohort's protid order.
    """

    name: str
    labels: np.ndarray
    block_id: str
    n_groups: int


@dataclass(frozen=True)
class FusionCohort:
    """Blocks over one protein index, plus what each was built to contain."""

    protids: list
    blocks: dict
    partitions: dict
    #: block_id -> the scale its values were generated at, for tests that want
    #: to state the incommensurability rather than rediscover it.
    scales: dict

    @property
    def n_proteins(self) -> int:
        return len(self.protids)

    def values(self, block_id: str) -> np.ndarray:
        return self.blocks[block_id]

    def separation(self, block_id: str, partition: str) -> float:
        """How visible a partition is in a block. See :func:`separation`."""
        return separation(self.blocks[block_id], self.partitions[partition].labels)

    def block_spec(self, block_id: str) -> BlockSpec:
        return BlockSpec(
            id=block_id,
            kind="features",
            fusable=True,
            metric="euclidean",
            normalization="none",
            provider="fusion_cohort",
        )

    def block_result(self, block_id: str) -> BlockResult:
        """A real :class:`BlockResult`, so tests can drive the real path.

        Group 7b's third defect was a fixture that wrote a table shape the
        pipeline never produces; 36 unit tests passed and the entry point
        crashed on the first run. Handing back the same type the store hands
        back is the cheap half of not repeating that.
        """
        return BlockResult(
            spec=self.block_spec(block_id),
            protids=list(self.protids),
            features=np.asarray(self.blocks[block_id], dtype=np.float64),
        )


def separation(values: np.ndarray, labels: np.ndarray) -> float:
    """Mean between-group distance over mean within-group distance.

    1.0 means the grouping is invisible in these features; above 1 means the
    groups are further from each other than from themselves. It is deliberately
    a ratio rather than a raw distance, because the blocks here differ in scale
    by four orders of magnitude and a raw distance would compare units.

    Computed by explicit broadcasting rather than by the Gram identity used in
    production code: this is the fixture's own measuring stick, and a measuring
    stick that shares an implementation with the thing it measures cannot catch
    that implementation being wrong.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    n = values.shape[0]
    distances = np.sqrt(((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2))
    same = labels[:, None] == labels[None, :]
    off_diagonal = ~np.eye(n, dtype=bool)
    within = distances[same & off_diagonal]
    between = distances[~same]
    if within.size == 0 or between.size == 0:
        raise ValueError("separation needs at least two groups and a repeated label")
    within_mean = float(within.mean())
    if within_mean == 0.0:
        return float("inf") if float(between.mean()) > 0 else 1.0
    return float(between.mean()) / within_mean


def _crossed_labels(n: int, rng: np.random.RandomState) -> tuple:
    """Two exactly-balanced, exactly-independent partitions of ``n`` proteins.

    Built as the full cross-product repeated ``n / (FOLD_GROUPS *
    CHEMISTRY_GROUPS)`` times and then permuted, rather than by drawing each
    label independently. Independent draws are independent only in expectation;
    at N=240 they leave a contingency table lopsided enough that a fusion which
    recovered `fold` would also score above chance on `chemistry`, and the test
    asserting the second is null would be asserting the sample rather than the
    construction.
    """
    cell = FOLD_GROUPS * CHEMISTRY_GROUPS
    repeats = n // cell
    fold = np.tile(np.repeat(np.arange(FOLD_GROUPS), CHEMISTRY_GROUPS), repeats)
    chemistry = np.tile(np.tile(np.arange(CHEMISTRY_GROUPS), FOLD_GROUPS), repeats)
    order = rng.permutation(n)
    return fold[order], chemistry[order]


def _structured(labels, n_groups, shape, rng) -> np.ndarray:
    """One centroid per group, plus isotropic noise."""
    n_columns, centroid_scale, noise_scale = shape
    centroids = rng.normal(0.0, centroid_scale, size=(n_groups, n_columns))
    return centroids[labels] + rng.normal(0.0, noise_scale, size=(len(labels), n_columns))


def fusion_cohort(*, n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> FusionCohort:
    """Generate the cohort. Deterministic in ``seed`` alone."""
    cell = FOLD_GROUPS * CHEMISTRY_GROUPS
    if n % cell:
        raise ValueError(
            f"n must be a multiple of {cell} so that the two partitions cross exactly, "
            f"got {n}. An inexact crossing makes them weakly dependent, and that "
            "dependence is indistinguishable from fusion working."
        )
    rng = np.random.RandomState(seed)
    protids = [f"FC{i:04d}" for i in range(n)]

    fold, chemistry = _crossed_labels(n, rng)
    wide = _structured(fold, FOLD_GROUPS, WIDE_SHAPE, rng)
    narrow = _structured(chemistry, CHEMISTRY_GROUPS, NARROW_SHAPE, rng)

    noise = rng.normal(0.0, 1.0, size=(n, 12))

    # Every row identical, so every pairwise distance is exactly zero.
    degenerate = np.tile(rng.normal(0.0, 1.0, size=(1, 3)), (n, 1))

    # Ordinary except for column 1, which never varies.
    constant_column = rng.normal(0.0, 1.0, size=(n, 5))
    constant_column[:, 1] = 7.0

    blocks = {
        WIDE_BLOCK: wide,
        NARROW_BLOCK: narrow,
        RESCALED_BLOCK: narrow * RESCALE_FACTOR,
        NOISE_BLOCK: noise,
        DEGENERATE_BLOCK: degenerate,
        CONSTANT_COLUMN_BLOCK: constant_column,
    }
    partitions = {
        "fold": PlantedPartition("fold", fold, WIDE_BLOCK, FOLD_GROUPS),
        "chemistry": PlantedPartition("chemistry", chemistry, NARROW_BLOCK, CHEMISTRY_GROUPS),
    }
    scales = {
        WIDE_BLOCK: WIDE_SHAPE[1],
        NARROW_BLOCK: NARROW_SHAPE[1],
        RESCALED_BLOCK: NARROW_SHAPE[1] * RESCALE_FACTOR,
        NOISE_BLOCK: 1.0,
        DEGENERATE_BLOCK: 0.0,
        CONSTANT_COLUMN_BLOCK: 1.0,
    }
    return FusionCohort(protids=protids, blocks=blocks, partitions=partitions, scales=scales)


def write_fusion_cohort(root, cohort: FusionCohort, block_ids=None) -> list:
    """Persist blocks through the real :class:`~spaces.store.BlockStore`.

    Returns the directories written. Tests that exercise an entry point need
    blocks on disk in the form the entry point reads, not a hand-built
    directory that resembles one.
    """
    from spaces.store import BlockStore

    store = BlockStore(str(root))
    return [store.write_block(cohort.block_result(b)) for b in (block_ids or cohort.blocks)]
