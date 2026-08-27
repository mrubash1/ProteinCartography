#!/usr/bin/env python
"""Do two blocks carry different information, or the same information twice?

ADR 0002 promises that no fused map exists without its contribution shares
beside it, and group 8a delivered them. But a share answers "how much of the
arithmetic came from this block", and the question a reader actually has is
"how much did adding this block change what I am looking at". Those come apart
exactly when two blocks are redundant:

    Two blocks whose pairwise distances correlate at 0.98 produce a fused map
    that is indistinguishable from either block alone. The shares still report
    50/50, and that number is *correct* -- it describes where the numbers came
    from. It is also useless, because the same map would have appeared with one
    of the blocks deleted.

There is nothing in a contribution share that can reveal this, which is why it
needs its own number and why Phase 5 asks for it to be printed *before* any
fusion runs. It is also the concrete form of a suspicion the domain already
has: protein language model embeddings encode a great deal of structure, and
aggregate biophysical descriptors are substantially hydrophobicity, so two
blocks that sound independent frequently are not.

**The comparison is between distance matrices, not between feature matrices.**
Blocks have different dimensionalities -- 200 columns against 4 in the fixture,
thousands against a handful in practice -- so their features cannot be
correlated column-wise at all. What they share is a protein index, so what can
be compared is what each says about every *pair* of proteins.

Both a Pearson and a Spearman correlation are reported, because they answer
different questions and their disagreement is informative. Pearson asks whether
the two blocks agree about how far apart things are; Spearman asks only whether
they agree about the ordering. A pair that is high on Spearman and low on
Pearson agrees about which proteins are similar and disagrees about the scale,
which is a block pair worth fusing.

Normalization is irrelevant here and that is load-bearing rather than
incidental: both correlations are invariant under multiplying either block's
distances by a positive constant, so the answer does not depend on ADR 0002's
unit-mean-distance step having run. That is what lets this diagnostic report
before a fusion strategy has been chosen. The fixture pins it -- `narrow` and
`narrow_rescaled` differ by a factor of 1000 and correlate at exactly 1.
"""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from coregistration import average_ranks, pairwise_distances

__all__ = [
    "REDUNDANT_THRESHOLD",
    "BlockPair",
    "RedundancyReport",
    "correlate_distances",
    "redundancy",
]

#: At and above this a pair is called redundant. A reporting convention, not a
#: result: there is no threshold at which two representations become "the
#: same", and the number exists so that the warning fires somewhere nameable
#: rather than being left to the reader's eye. Chosen high enough that ordinary
#: agreement between two real representations does not trip it.
REDUNDANT_THRESHOLD = 0.90


class RedundancyError(ValueError):
    """Raised when two blocks cannot be compared at all."""


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Every unordered pair once. The diagonal is structurally zero in both
    blocks and would inflate any correlation computed over it."""
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt(float(a @ a) * float(b @ b))
    # A block with one distinct distance -- every protein equidistant from
    # every other -- has no variation to correlate. NaN rather than 0: the two
    # blocks did not disagree, the question was not answerable.
    #
    # float() on the whole quotient, not on the numerator. `np.sqrt` returns a
    # np.float64, so dividing a Python float by it gives a np.float64 back, and
    # `np.float64 >= threshold` is a np.bool_ that `json.dump` refuses. The
    # report goes into a manifest, so that surfaces as a crashed rule rather
    # than as a wrong number.
    return float(a @ b / denominator) if denominator > 0 else float("nan")


def correlate_distances(values_a: np.ndarray, values_b: np.ndarray) -> tuple:
    """``(pearson, spearman)`` between two blocks' pairwise distances.

    Args:
        values_a: an ``(N, D_a)`` feature matrix.
        values_b: an ``(N, D_b)`` feature matrix over the same proteins in the
            same order.

    Spearman goes through :func:`coregistration.average_ranks`, so tied
    distances share their mean rank. Tie handling is not a nicety here: a
    censored TM-score block has exact zeros in their thousands, and ranking
    those by position would manufacture an ordering out of the order the file
    happened to be written in, which both blocks would then "agree" on.
    """
    a, b = np.asarray(values_a, dtype=np.float64), np.asarray(values_b, dtype=np.float64)
    if a.shape[0] != b.shape[0]:
        raise RedundancyError(
            f"the two blocks cover {a.shape[0]} and {b.shape[0]} proteins. They must "
            "be aligned to a shared index before their distances can be compared."
        )
    if a.shape[0] < 3:
        raise RedundancyError(
            f"correlating pairwise distances over {a.shape[0]} proteins uses "
            f"{a.shape[0] * (a.shape[0] - 1) // 2} pair(s), which is too few to mean "
            "anything. At least 3 proteins are needed."
        )
    distances_a = _upper_triangle(pairwise_distances(a))
    distances_b = _upper_triangle(pairwise_distances(b))
    return (
        _pearson(distances_a, distances_b),
        _pearson(average_ranks(distances_a), average_ranks(distances_b)),
    )


@dataclass(frozen=True)
class BlockPair:
    """How much two blocks agree about the same proteins."""

    block_a: str
    block_b: str
    pearson: float
    spearman: float
    n_pairs: int

    @property
    def redundant(self) -> bool:
        return self.spearman == self.spearman and self.spearman >= REDUNDANT_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "block_a": self.block_a,
            "block_b": self.block_b,
            "pearson": self.pearson,
            "spearman": self.spearman,
            "n_pairs": self.n_pairs,
            "redundant": self.redundant,
        }


@dataclass(frozen=True)
class RedundancyReport:
    """Every pair of blocks in one space, and what fusing them would buy."""

    block_ids: tuple
    pairs: tuple

    def pair(self, block_a: str, block_b: str) -> BlockPair:
        for candidate in self.pairs:
            if {candidate.block_a, candidate.block_b} == {block_a, block_b}:
                return candidate
        raise KeyError(f"no pair {block_a!r} / {block_b!r} in this report")

    @property
    def most_redundant(self) -> BlockPair | None:
        scored = [p for p in self.pairs if p.spearman == p.spearman]
        return max(scored, key=lambda p: p.spearman) if scored else None

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame([p.to_dict() for p in self.pairs])

    def to_dict(self) -> dict:
        return {
            "block_ids": list(self.block_ids),
            "pairs": [p.to_dict() for p in self.pairs],
            "warnings": self.warnings(),
        }

    def warnings(self) -> list:
        notes = []
        for pair in self.pairs:
            if pair.redundant:
                notes.append(
                    f"blocks {pair.block_a!r} and {pair.block_b!r} correlate at "
                    f"{pair.spearman:.2f} (Spearman) over their pairwise distances, so "
                    "they carry substantially the same information. Fusing them does "
                    "not combine two views of these proteins; it counts one view "
                    "twice. The contribution shares will still divide the weight "
                    "between them, and that split describes the arithmetic rather "
                    "than the evidence."
                )
            elif pair.spearman != pair.spearman:
                notes.append(
                    f"blocks {pair.block_a!r} and {pair.block_b!r} could not be "
                    "correlated: one of them gives every pair of proteins the same "
                    "distance, so it has no ordering to compare. That block carries "
                    "no geometry and will contribute nothing to a fused map."
                )
            elif abs(pair.pearson - pair.spearman) >= 0.25:
                notes.append(
                    f"blocks {pair.block_a!r} and {pair.block_b!r} agree about the "
                    f"ordering of distances more than about their size (Spearman "
                    f"{pair.spearman:.2f}, Pearson {pair.pearson:.2f}). They rank the "
                    "same proteins as similar on different scales, which is the case "
                    "the unit-mean-distance normalization exists for."
                )
        if not notes:
            notes.append(
                f"no pair of the {len(self.block_ids)} blocks correlates at or above "
                f"{REDUNDANT_THRESHOLD}; each is contributing something the others "
                "do not."
            )
        return notes


def redundancy(blocks: Mapping) -> RedundancyReport:
    """Every unordered pair of blocks, in the order the blocks were given.

    Args:
        blocks: block id to its ``(N, D)`` feature matrix. Every block must
            cover the same proteins in the same order.
    """
    block_ids = list(blocks)
    if len(block_ids) < 2:
        raise RedundancyError(
            f"redundancy compares blocks against each other, and {len(block_ids)} "
            "block(s) makes no pair. A single-block space has nothing to be "
            "redundant with."
        )
    sizes = {block_id: np.asarray(values).shape[0] for block_id, values in blocks.items()}
    if len(set(sizes.values())) != 1:
        raise RedundancyError(
            f"the blocks cover different numbers of proteins: {sizes}. Align them to "
            "a shared index first."
        )
    n = next(iter(sizes.values()))

    pairs = []
    for i, block_a in enumerate(block_ids):
        for block_b in block_ids[i + 1 :]:
            pearson, spearman = correlate_distances(blocks[block_a], blocks[block_b])
            pairs.append(BlockPair(block_a, block_b, pearson, spearman, n_pairs=n * (n - 1) // 2))
    return RedundancyReport(block_ids=tuple(block_ids), pairs=tuple(pairs))
