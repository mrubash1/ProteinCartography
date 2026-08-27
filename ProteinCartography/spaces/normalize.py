"""Within-block normalization, applied before a block enters a geometry.

`BlockSpec.normalization` has been recorded on every block since the schema
existed and applied by nothing (FOLLOWUPS #32, the same shape as #29 for
`spec.metric`). This module is what applies it.

WHY IT IS NOT THE SAME THING AS FUSION'S NORMALIZATION, which is the distinction
that kept the defect alive. `fusion.py` normalizes each block to unit mean
distance before weighting, "always, with no way to opt out" -- that is ADR 0002's
BETWEEN-block contract, and it makes a weight of 1.0 against 1.0 mean something.
It is a single scalar per block, so it cannot change the relative scale of the
COLUMNS inside a block. `zscore_within` can, and that is the whole difference:

    physicochemistry on the production actin cohort reported
    "made of: isoelectric_point 97.1% - gravy 2.9% - charge_per_residue 0.0%",
    not because isoelectric point carries the signal but because it runs 4 to 12
    while gravy runs about -2 to 2 and charge per residue sits near zero. The
    columns entered the distance raw. `biophys` declared `zscore_within` the
    whole time.

ORDER OF OPERATIONS, recorded here because PC-011 found it written down nowhere:
**within-block first, then fusion's between-block scaling.** That order leaves
`fuse_late` invariant for a block that declares `unit_mean_distance` -- the
block arrives at unit mean distance and fusion's own scaling is then a no-op --
whereas the reverse double-normalizes.

Numpy only, no scipy: ADR 0006 rule 3 requires the framework to work with zero
optional dependencies installed, and this runs inside `reduce_space` on the
default path.
"""

from __future__ import annotations

import numpy as np

#: Vocabularies live in `spaces.base`; this maps them to behaviour.
__all__ = ["normalize_block", "mean_pairwise_distance"]


def mean_pairwise_distance(values: np.ndarray, chunk: int = 512) -> float:
    """Mean Euclidean distance over all unordered pairs of rows.

    Chunked rather than materialising the full (N, N) matrix, and computed
    through the Gram identity rather than by subtracting rows. This runs on the
    path ADR 0006 requires to work with no scipy, so
    `scipy.spatial.distance.pdist` is not available to call.

    **The identity is not an optimisation, it is what makes this usable at all.**
    The obvious form, ``block[:, None, :] - values[None, :, :]``, materialises a
    ``(chunk, N, D)`` intermediate. For most blocks D is small and that is fine.
    For the `tmscore` block with ``representation: profile`` -- the pipeline's
    own representation, where each protein IS its row of the similarity matrix --
    D equals N, so the intermediate is ``chunk * N**2``. Measured: 938 MB peak at
    N=600, and **29.9 GB at the production cohort size of 2,703**. Nothing here
    ran before PC-011, because `unit_mean_distance` was declared and never
    applied; honoring the field put this on the production path.

    ``|a - b|^2 = |a|^2 + |b|^2 - 2 a.b`` needs only the ``(chunk, N)`` inner
    products, which is 11 MB at N=2,703 instead of 29.9 GB, and hands the work
    to BLAS besides.

    Returns 0.0 for fewer than two rows, which the caller must treat as "no
    scale to normalize" rather than dividing by it.
    """
    n = values.shape[0]
    if n < 2:
        return 0.0
    # CENTRE FIRST. The identity subtracts two large numbers to get a small
    # one, so it loses relative accuracy exactly when the rows are tightly
    # clustered far from the origin -- measured at 1e-5 relative error on rows
    # spread by 1e-3 around 500. A pairwise distance is translation-invariant,
    # so removing the column mean changes no distance and leaves the norms on
    # the order of the spread instead of the offset. That takes the same case
    # to 1e-12.
    array = np.asarray(values, dtype=np.float64)
    array = array - array.mean(axis=0)
    square_norms = np.einsum("ij,ij->i", array, array)
    total = 0.0
    count = 0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = array[start:stop]
        squared = square_norms[start:stop, None] + square_norms[None, :] - 2.0 * (block @ array.T)
        # A squared distance is never negative; rounding in the identity can
        # make one very slightly so, and `sqrt` would return nan rather than 0.
        np.maximum(squared, 0.0, out=squared)
        dist = np.sqrt(squared)
        # Only the strictly-upper triangle, accumulated blockwise.
        rows = np.arange(start, stop)[:, None]
        cols = np.arange(n)[None, :]
        mask = cols > rows
        total += float(dist[mask].sum())
        count += int(mask.sum())
    return total / count if count else 0.0


def normalize_block(values: np.ndarray, rule: str, *, what: str = "block") -> np.ndarray:
    """`values` rescaled according to `rule`, as a float64 copy.

    `what` names the block in any error, because a caller hitting this has a
    config in front of them and not this file.
    """
    if rule == "none":
        # The array itself, not a promoted copy. `test_a_single_block_space_is_
        # unchanged_by_going_through_fusion` exists to catch exactly that: the
        # store holds float32 and a silent promotion to float64 doubles every
        # block on the way to the reducer. A first draft of this module did
        # promote here, and that test caught it.
        return values

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{what}: normalization needs an (N, D) matrix, got shape {array.shape}")

    if rule == "zscore_within":
        # Per COLUMN. A constant column has no scale to normalize and its
        # z-score is undefined; it becomes zeros, which contributes nothing to
        # any distance rather than contributing a division by zero. That is the
        # honest reading: a descriptor identical across the cohort carries no
        # information about which proteins are alike.
        mean = array.mean(axis=0)
        sd = array.std(axis=0)
        out = np.zeros_like(array)
        varies = sd > 0
        out[:, varies] = (array[:, varies] - mean[varies]) / sd[varies]
        # Computed in float64 for the accumulation, returned in the store's
        # dtype. See the note under "none" above.
        return out.astype(values.dtype, copy=False)

    if rule == "unit_mean_distance":
        scale = mean_pairwise_distance(array)
        if scale <= 0:
            # Every row identical, or a single row. Scaling is undefined and the
            # block carries no shape; returning it unchanged is the only option
            # that does not invent one.
            return values
        return (array / scale).astype(values.dtype, copy=False)

    raise ValueError(f"{what}: unknown normalization {rule!r}")
