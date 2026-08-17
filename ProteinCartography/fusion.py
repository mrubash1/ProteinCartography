#!/usr/bin/env python
"""Combining several blocks into one geometry, with the shares written down.

Fusion is not the default product of this work -- co-registration is (ADR 0001),
because fusing first destroys the cross-space disagreement that is the most
interesting thing several representations produce. Fusion is what you invoke
when you want one map anyway, and ADR 0002 is the contract it implements:

* **Block scale is normalized before weighting, always, with no way to opt
  out.** Each block's distances are divided by their mean, so every block
  arrives at unit mean distance and a weight of 1.0 against 1.0 means what a
  reader assumes it means. Without this, a weight vector means something
  different in every config, because it is competing with whatever units the
  block happens to be in.
* **Contribution share is computed, not asserted.** Every strategy reports what
  fraction of the fused geometry each block actually supplied, the shares are
  checked to sum to 1, and a block above 70% triggers a warning that says the
  map is that block's map.

Three things this module learned by being written, all of which changed ADR 0002
rather than being quietly worked around:

1. **The nominal share and the realized share are different numbers.** ADR 0002
   defines `late`'s share as ``w_i * mean(d̃_i)² / Σ w_j * mean(d̃_j)²``, and
   since normalization makes every ``mean(d̃)`` exactly 1, that formula returns
   the normalized weight vector -- it cannot tell you anything the config did
   not already say. What a block actually contributes to a squared fused
   distance is ``w_i * mean(d̃_i²)``, and ``mean(d̃²) = 1 + var(d̃)``, so a block
   whose distances are widely dispersed contributes more than its weight. Both
   are reported. They agree only when the blocks have equal dispersion.
2. **The published SNF kernel is not scale-invariant**, because its exponent is
   ``d² / (μ·ε)`` with ``ε`` in units of distance -- so the whole expression
   scales with the block, and multiplying a block by 1000 changes its affinities
   entirely. Applying ADR 0002's normalization contract *before* the kernel
   fixes this, and is required by the contract anyway. `graph` therefore
   normalizes first, and is scale-invariant where the published form is not.
3. **`graph` is where per-protein weights become real.** `SpaceSpec.weights` is
   one scalar per block, so the per-protein weighting ADR 0002 names as SNF's
   unique advantage had nowhere to live (FOLLOWUPS #16). It lives on the
   *output* rather than the input: SNF's fused affinity gives every protein its
   own mixture of the blocks, and `BlockContribution.per_protein_share` records
   it. A protein whose structure is well determined and whose chemistry is
   uninformative is weighted accordingly, without anyone configuring it.

Everything here is numpy. SNF has a package (`snfpy`) and standardization has a
scikit-learn transformer, and ADR 0006 requires the default configuration to run
with neither installed.

This module computes and does not do I/O. :mod:`reduce_space` is the entry
point, exactly as :mod:`enrichment` is computation and :mod:`enrich_clusters` is
the entry point.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

# Shared rather than reimplemented, for the reason the reducer core is shared:
# two implementations of the pipeline's euclidean distance would eventually
# disagree, and the disagreement would look like a scientific result. The Gram
# identity there also matters here -- broadcasting the differences for a
# 2500-protein profile block needs hundreds of gigabytes of intermediates.
from coregistration import pairwise_distances

__all__ = [
    "DOMINANCE_THRESHOLD",
    "FUSION_STRATEGIES",
    "BlockContribution",
    "FusionError",
    "FusionInput",
    "FusionResult",
    "fuse",
    "fuse_early",
    "fuse_graph",
    "fuse_late",
    "fuse_none",
    "standardize",
    "unit_mean_distance",
]

#: The strategies this module implements. `spaces.base.STRATEGIES` also carries
#: `coregistered`, which is the *absence* of fusion -- N independent spaces --
#: and so is handled by `coregister.py`, not here.
FUSION_STRATEGIES = ("none", "early", "late", "graph")

#: Above this share, a single block is the map and the output should say so.
DOMINANCE_THRESHOLD = 0.70

#: Shares are checked, not assumed (ADR 0002). The tolerance is loose enough for
#: an N=5000 accumulation and tight enough that a missing term cannot hide.
SHARE_TOLERANCE = 1e-9

# SNF defaults, from Wang et al. 2014 (Nature Methods 11:333). `mu` is their
# hyperparameter, recommended in [0.3, 0.8]; `k` is the neighborhood size and
# `iterations` the number of diffusion steps, recommended 10-20.
DEFAULT_GRAPH_K = 20
DEFAULT_GRAPH_MU = 0.5
DEFAULT_GRAPH_ITERATIONS = 20


class FusionError(ValueError):
    """Raised when blocks cannot be fused into a geometry."""


@dataclass(frozen=True)
class FusionInput:
    """One block's feature matrix and its configured weight.

    Rows must already be aligned across every input: fusion combines row `i` of
    each block as one protein, and nothing here can check that claim. The caller
    aligns -- :mod:`reduce_space` does it through
    :meth:`index.ProteinIndex.align` -- and this module states the requirement
    rather than pretending to verify it.
    """

    block_id: str
    values: np.ndarray
    weight: float = 1.0

    def __post_init__(self):
        # Floating-point input is kept at its own precision rather than promoted.
        # The block store writes float32 deliberately (ADR 0004), and `none`
        # hands its block straight to the reducer -- promoting here would move
        # every existing single-block embedding as a side effect of adding
        # fusion. The strategies that form distances upcast internally, in
        # `pairwise_distances` and `standardize`, so nothing computes in float32.
        values = np.asarray(self.values)
        if not np.issubdtype(values.dtype, np.floating):
            values = values.astype(np.float64)
        if values.ndim != 2:
            raise FusionError(
                f"block {self.block_id!r}: expected an (N, D) feature matrix, got shape "
                f"{values.shape}."
            )
        if not np.isfinite(values).all():
            raise FusionError(
                f"block {self.block_id!r}: contains NaN or infinity. A block reaches "
                "fusion only after `BlockResult` has refused any NaN not covered by a "
                "channel, so this is either an unmasked hole or an overflow in the "
                "provider."
            )
        if self.weight < 0:
            raise FusionError(
                f"block {self.block_id!r}: a negative weight ({self.weight}) has no "
                "meaning -- distances cannot be subtracted from one another."
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "weight", float(self.weight))

    @property
    def n_proteins(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True)
class BlockContribution:
    """What one block supplied to the fused geometry.

    `share` is the strategy's own definition, and `realized_share` is the
    fraction of the fused quantity the block actually accounts for. For `late`
    they differ whenever the blocks differ in distance dispersion; see the
    module docstring. For `early` they are the same number, because there the
    definition already is the realized variance.
    """

    block_id: str
    weight: float
    n_features: int
    share: float
    realized_share: float
    #: Mean pairwise distance before normalization. The number that says how far
    #: apart two blocks' units were, and `None` for `early`, which never forms a
    #: distance.
    mean_distance: float | None = None
    #: Spread of the normalized distances, `var(d̃)`. Zero means every pair is
    #: equally far apart, in which case the block contributes scale and no shape.
    dispersion: float | None = None
    #: One share per protein, for strategies that produce them. `graph` does;
    #: `early` and `late` apply one weight to every protein by construction, and
    #: report `None` rather than an array of identical numbers pretending
    #: otherwise.
    per_protein_share: np.ndarray | None = None

    def to_dict(self) -> dict:
        out = {
            "block_id": self.block_id,
            "weight": self.weight,
            "n_features": self.n_features,
            "share": self.share,
            "realized_share": self.realized_share,
            "mean_distance": self.mean_distance,
            "dispersion": self.dispersion,
        }
        if self.per_protein_share is not None:
            shares = np.asarray(self.per_protein_share, dtype=float)
            out["per_protein_share"] = {
                "min": float(shares.min()),
                "median": float(np.median(shares)),
                "max": float(shares.max()),
            }
        return out


@dataclass(frozen=True)
class FusionResult:
    """The fused values, plus who supplied them.

    `values` is always an ``(N, D)`` matrix the reducer core consumes directly,
    whatever the strategy formed on the way. `representation` says what its
    columns are, because that is not recoverable from the array:

    * ``features`` -- real features, from `none` and `early`.
    * ``distance_profile`` -- row `i` is protein `i`'s distance to every
      protein, from `late`.
    * ``affinity_profile`` -- row `i` is protein `i`'s fused affinity to every
      protein, from `graph`.

    The last two are the representation the pipeline has always used for the
    TM-score matrix (`representation: profile`), reached from a different
    direction. Using it here rather than inventing a metric-aware reducer keeps
    every space going through one reducer core, which is the property that stops
    two PCAs drifting apart.
    """

    values: np.ndarray
    strategy: str
    representation: str
    contributions: tuple
    warnings: tuple = ()
    params_used: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.contributions:
            raise FusionError("a fusion result must name at least one contributing block")
        # ADR 0002: "Shares must sum to 1 under every strategy; this is asserted,
        # not assumed." A share that does not sum is a share computed from a
        # different denominator than the one the geometry used, and it would be
        # written to the manifest and read as evidence.
        for name in ("share", "realized_share"):
            total = sum(getattr(c, name) for c in self.contributions)
            if abs(total - 1.0) > SHARE_TOLERANCE:
                raise FusionError(
                    f"{self.strategy}: {name} over "
                    f"{[c.block_id for c in self.contributions]} sums to {total!r}, "
                    "not 1. A share that does not sum describes a geometry other "
                    "than the one that was built."
                )

    @property
    def shares(self) -> dict:
        return {c.block_id: c.share for c in self.contributions}

    @property
    def realized_shares(self) -> dict:
        return {c.block_id: c.realized_share for c in self.contributions}

    @property
    def dominant(self) -> BlockContribution | None:
        """The block above :data:`DOMINANCE_THRESHOLD`, if there is one."""
        for c in self.contributions:
            if c.realized_share > DOMINANCE_THRESHOLD:
                return c
        return None

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "representation": self.representation,
            "contributions": [c.to_dict() for c in self.contributions],
            "warnings": list(self.warnings),
            "params_used": dict(self.params_used),
        }

    def describe(self) -> str:
        """The weight vector and the shares, for stderr and for the log.

        ADR 0002 calls the weight vector "a displayed object, not a buried
        config value". This is the display for a run without a browser.
        """
        lines = [f"fusion strategy {self.strategy!r} over {len(self.contributions)} block(s):"]
        for c in sorted(self.contributions, key=lambda c: -c.realized_share):
            detail = f"D={c.n_features}"
            if c.mean_distance is not None:
                detail += f", mean distance {c.mean_distance:.4g}"
            lines.append(
                f"  {c.block_id}: weight {c.weight:g}, share {c.share:.1%}, "
                f"realized {c.realized_share:.1%} ({detail})"
            )
        lines.extend(f"  WARNING: {w}" for w in self.warnings)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# the normalization contract
# ---------------------------------------------------------------------------


def _off_diagonal(matrix: np.ndarray) -> np.ndarray:
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def unit_mean_distance(distances: np.ndarray, block_id: str = "?") -> np.ndarray:
    """Scale a distance matrix so its mean off-diagonal entry is exactly 1.

    The whole of ADR 0002's normalization contract, in one function. The
    diagonal is excluded from the mean because it is structurally zero and
    including it would make the normalization depend on N.

    Raises rather than dividing by zero. A block whose every distance is zero --
    one protein, or a provider that returned a constant for all of them --
    carries no geometry at all, and the alternative to this error is a matrix of
    NaN coordinates that reaches a plot.
    """
    distances = np.asarray(distances, dtype=np.float64)
    if distances.shape[0] < 2:
        raise FusionError(
            f"block {block_id!r}: fusion needs at least two proteins, got " f"{distances.shape[0]}."
        )
    mean = float(_off_diagonal(distances).mean())
    if mean <= 0.0:
        raise FusionError(
            f"block {block_id!r}: every pairwise distance is zero, so its mean "
            "distance is zero and the block cannot be normalized to unit mean "
            "distance (ADR 0002). Every protein has identical values in this "
            "block, which means it carries no geometry -- drop it from the space, "
            "or find out why the provider returned a constant."
        )
    return distances / mean


def standardize(values: np.ndarray, block_id: str = "?") -> tuple:
    """Z-score each column. Returns ``(standardized, n_constant_columns)``.

    A column with zero variance is left at zero rather than divided by it. It
    contributes nothing either way -- every protein has the same value -- but
    the count is returned so the caller can say so, because the alternative is a
    block whose reported dimensionality is larger than the number of dimensions
    it actually spans.
    """
    values = np.asarray(values, dtype=np.float64)
    means = values.mean(axis=0)
    deviations = values.std(axis=0)
    constant = deviations == 0
    if constant.all():
        raise FusionError(
            f"block {block_id!r}: every column is constant, so standardizing it "
            "leaves nothing. The block carries no geometry."
        )
    safe = np.where(constant, 1.0, deviations)
    standardized = (values - means) / safe
    standardized[:, constant] = 0.0
    return standardized, int(constant.sum())


def _check_inputs(inputs, strategy: str, minimum: int = 1) -> tuple:
    inputs = tuple(inputs)
    if len(inputs) < minimum:
        raise FusionError(
            f"strategy {strategy!r} needs at least {minimum} block(s), got {len(inputs)}."
        )
    ids = [i.block_id for i in inputs]
    if len(set(ids)) != len(ids):
        raise FusionError(f"strategy {strategy!r}: a block is listed more than once: {ids}")
    sizes = {i.n_proteins for i in inputs}
    if len(sizes) != 1:
        per_block = {i.block_id: i.n_proteins for i in inputs}
        raise FusionError(
            "every block must cover the same proteins in the same order, but the "
            f"blocks have {sorted(sizes)} rows: {per_block}. Align them to a shared "
            "index before fusing."
        )
    total_weight = sum(i.weight for i in inputs)
    if total_weight <= 0:
        raise FusionError(
            f"strategy {strategy!r}: every weight is zero, so no block contributes "
            "anything and the fused geometry would be empty."
        )
    return inputs


def _dominance_warning(contributions) -> list:
    top = max(contributions, key=lambda c: c.realized_share)
    if top.realized_share <= DOMINANCE_THRESHOLD:
        return []
    return [
        f"block {top.block_id!r} accounts for {top.realized_share:.1%} of the fused "
        f"geometry, above the {DOMINANCE_THRESHOLD:.0%} threshold. This is that "
        "block's map; the other blocks are perturbing it, not shaping it."
    ]


# ---------------------------------------------------------------------------
# the strategies
# ---------------------------------------------------------------------------


def fuse_none(inputs) -> FusionResult:
    """One block, unchanged. The identity, kept so the caller has no special case."""
    inputs = _check_inputs(inputs, "none")
    if len(inputs) != 1:
        raise FusionError(
            f"strategy 'none' means a single block, but {len(inputs)} were given: "
            f"{[i.block_id for i in inputs]}. Choose a fusion strategy, or split "
            "these into separate co-registered spaces."
        )
    only = inputs[0]
    return FusionResult(
        values=only.values,
        strategy="none",
        representation="features",
        contributions=(
            BlockContribution(
                block_id=only.block_id,
                weight=only.weight,
                n_features=only.n_features,
                share=1.0,
                realized_share=1.0,
            ),
        ),
        params_used={"strategy": "none"},
    )


def fuse_early(inputs) -> FusionResult:
    """Standardize per block, weight, concatenate. One feature space, one PCA.

    The standard baseline, and never a default. Standardizing puts every column
    at variance 1, so a block contributes variance in proportion to its *column
    count* -- which is exactly why `early` hands a 200-column block 98% of a
    geometry it shares with a 4-column one, however carefully the weights were
    chosen. The warning says so with the run's own numbers rather than in the
    abstract.

    Weighting is by ``sqrt(w)`` on the values, because variance is quadratic in
    the values and the weight is meant to act on variance.
    """
    inputs = _check_inputs(inputs, "early")
    total_weight = sum(i.weight for i in inputs)

    columns = []
    variances = []
    constant_counts = {}
    for block in inputs:
        standardized, n_constant = standardize(block.values, block.block_id)
        scaled = standardized * np.sqrt(block.weight)
        columns.append(scaled)
        variances.append(float(scaled.var(axis=0).sum()))
        constant_counts[block.block_id] = n_constant

    total_variance = float(sum(variances))
    if total_variance <= 0:
        raise FusionError(
            "early fusion: the standardized blocks carry no variance at all, so "
            "there is no geometry to reduce."
        )

    contributions = tuple(
        BlockContribution(
            block_id=block.block_id,
            weight=block.weight,
            n_features=block.n_features,
            # For `early` the definition *is* the realized variance, so the two
            # shares coincide. Reporting both keeps the manifest one shape.
            share=variance / total_variance,
            realized_share=variance / total_variance,
        )
        for block, variance in zip(inputs, variances)
    )

    warnings = []
    if len(inputs) > 1:
        ranked = sorted(contributions, key=lambda c: -c.realized_share)
        top, bottom = ranked[0], ranked[-1]
        warnings.append(
            f"block {top.block_id!r} (D={top.n_features}) contributes "
            f"{top.realized_share:.1%} of total variance; block {bottom.block_id!r} "
            f"(D={bottom.n_features}) contributes {bottom.realized_share:.1%}. "
            "Standardizing equalizes the columns, not the blocks, so a block's "
            "share under `early` is essentially its column count. Consider "
            "`strategy: late`, which normalizes the blocks."
        )
    for block_id, n_constant in constant_counts.items():
        if n_constant:
            warnings.append(
                f"block {block_id!r} has {n_constant} constant column(s), which "
                "were left at zero rather than standardized. They occupy "
                "dimensions and contribute no variance."
            )

    return FusionResult(
        values=np.hstack(columns),
        strategy="early",
        representation="features",
        contributions=contributions,
        warnings=tuple(warnings),
        params_used={
            "strategy": "early",
            "normalization": "zscore_per_column",
            "weighting": "sqrt(w) on values",
            "total_weight": total_weight,
            "constant_columns": constant_counts,
        },
    )


def _normalized_distances(inputs) -> tuple:
    """Per block: unit-mean-distance normalized distances, and its raw mean."""
    normalized, means = [], []
    for block in inputs:
        distances = pairwise_distances(block.values)
        means.append(float(_off_diagonal(distances).mean()))
        normalized.append(unit_mean_distance(distances, block.block_id))
    return normalized, means


def fuse_late(inputs) -> FusionResult:
    r"""Combine in distance space: ``D² = Σ wᵢ·d̃ᵢ² / Σ wᵢ``.

    The recommended fusion path and the one that behaves sanely across wildly
    different D, because it compares blocks after each has been reduced to a
    distance and normalized to unit mean. Its cost is that feature-space
    structure is gone -- you can read off which *block* drove a distance, never
    which dimension.

    The division by ``Σ wᵢ``, which ADR 0002's formula omits, is a single global
    scale factor and changes no relative distance. It buys one property worth
    having: fusing a single block reproduces exactly that block's
    unit-mean-distance geometry, so `late` degenerates to the normalization
    rather than to the normalization times a constant that depends on how many
    blocks happened to be listed.
    """
    inputs = _check_inputs(inputs, "late")
    normalized, raw_means = _normalized_distances(inputs)
    total_weight = sum(i.weight for i in inputs)

    squared = np.zeros_like(normalized[0])
    for block, d in zip(inputs, normalized):
        squared += block.weight * (d**2)
    squared /= total_weight
    np.maximum(squared, 0.0, out=squared)
    np.fill_diagonal(squared, 0.0)
    fused = np.sqrt(squared)

    # ADR 0002's formula, computed from the measured means rather than from the
    # weights. mean(d̃) is 1 by construction, so this returns the normalized
    # weight vector -- and it returns something else the moment normalization
    # stops working, which is the only reason to compute it rather than state it.
    nominal = [
        block.weight * float(_off_diagonal(d).mean()) ** 2 for block, d in zip(inputs, normalized)
    ]
    nominal_total = float(sum(nominal))

    # What each block actually put into the fused squared distance. mean(d̃²) is
    # 1 + var(d̃), so a block with widely dispersed distances contributes more
    # than its weight, and a block whose pairs are all equally far apart
    # contributes exactly its weight and no shape.
    realized = [
        block.weight * float(_off_diagonal(d**2).mean()) for block, d in zip(inputs, normalized)
    ]
    realized_total = float(sum(realized))

    contributions = tuple(
        BlockContribution(
            block_id=block.block_id,
            weight=block.weight,
            n_features=block.n_features,
            share=nominal[i] / nominal_total,
            realized_share=realized[i] / realized_total,
            mean_distance=raw_means[i],
            dispersion=float(_off_diagonal(normalized[i]).var()),
        )
        for i, block in enumerate(inputs)
    )

    return FusionResult(
        values=fused,
        strategy="late",
        representation="distance_profile",
        contributions=contributions,
        warnings=tuple(_dominance_warning(contributions)),
        params_used={
            "strategy": "late",
            "normalization": "unit_mean_distance",
            "total_weight": total_weight,
        },
    )


# ---------------------------------------------------------------------------
# graph fusion (SNF)
# ---------------------------------------------------------------------------


def _scaled_exponential_kernel(distances: np.ndarray, k: int, mu: float) -> np.ndarray:
    """Wang et al. 2014 eq. 3: an affinity whose bandwidth is local to each pair.

    ``ε(i,j)`` is the average of `i`'s mean distance to its neighbors, `j`'s, and
    the distance between them, so a pair in a dense region is judged against a
    tighter scale than a pair in a sparse one. That local rescaling is the whole
    reason SNF handles blocks whose density varies across the cohort.

    The published kernel takes ``d²/(μ·ε)``, which has units of distance and is
    therefore *not* invariant to rescaling the block. Here it is invariant,
    because the distances arriving have already been normalized to unit mean by
    ADR 0002's contract -- which the contract requires regardless, and which
    this function depends on rather than re-doing.
    """
    n = distances.shape[0]
    neighbors = min(max(k, 2), n) - 1  # excluding self
    ordered = np.sort(distances + np.diag(np.full(n, np.inf)), axis=1)
    local_scale = ordered[:, :neighbors].mean(axis=1)
    epsilon = (local_scale[:, None] + local_scale[None, :] + distances) / 3.0
    # A pair can only reach zero here if both proteins are exact duplicates of
    # their whole neighborhood; the floor keeps that from becoming a NaN.
    epsilon = np.maximum(epsilon, np.finfo(float).tiny)
    affinity = np.exp(-(distances**2) / (mu * epsilon))
    return 0.5 * (affinity + affinity.T)


def _full_normalize(affinity: np.ndarray) -> np.ndarray:
    """Wang et al. eq. 8: rows sum to 1, with half the mass on the diagonal.

    Holding the diagonal at exactly 1/2 is what makes the diffusion converge --
    it stops the iteration draining all of a row's mass into its neighbors.

    The result is row-stochastic and, in general, *not* symmetric, because each
    row is divided by its own sum. That is the published form and it is kept:
    symmetrizing here would trade the row sums away at every step, and the row
    sums are what makes the per-step delta comparable across iterations. The
    fused output is symmetrized once, at the end, where affinity(i,j) ==
    affinity(j,i) is worth more than the row sums.
    """
    off = affinity.copy()
    np.fill_diagonal(off, 0.0)
    row_sums = off.sum(axis=1)
    isolated = row_sums <= 0
    safe = np.where(isolated, 1.0, row_sums)
    normalized = off / (2.0 * safe[:, None])
    np.fill_diagonal(normalized, 0.5)
    # A protein with no affinity to anything keeps all of its own mass rather
    # than half of it, so its row still sums to 1.
    if isolated.any():
        positions = np.flatnonzero(isolated)
        normalized[positions] = 0.0
        normalized[positions, positions] = 1.0
    return normalized


def _local_normalize(affinity: np.ndarray, k: int) -> np.ndarray:
    """Wang et al. eq. 9: keep each row's k strongest affinities, renormalize.

    Sparsifying to the k nearest is the second half of why SNF works: the
    strongest similarities are the reliable ones, and diffusing through the weak
    tail is diffusing through noise. `k` counts the protein itself, matching the
    reference implementation -- the diagonal is always a row's largest entry, so
    it is always among the kept.

    Ties at the cut keep *every* tied entry rather than an arbitrary k of them.
    That is deliberate: breaking a tie by column position would make the fused
    geometry depend on the order the proteins happened to be listed in, which is
    the failure ADR 0007 exists for, met in a different place.
    """
    n = affinity.shape[0]
    keep = min(max(k, 1), n)
    cut = np.partition(affinity, n - keep, axis=1)[:, n - keep][:, None]
    kept = np.where(affinity >= cut, affinity, 0.0)
    row_sums = kept.sum(axis=1)
    safe = np.where(row_sums <= 0, 1.0, row_sums)
    return kept / safe[:, None]


def fuse_graph(
    inputs,
    *,
    k: int = DEFAULT_GRAPH_K,
    mu: float = DEFAULT_GRAPH_MU,
    iterations: int = DEFAULT_GRAPH_ITERATIONS,
) -> FusionResult:
    """Similarity Network Fusion (Wang et al. 2014, Nature Methods 11:333).

    Each block becomes an affinity network; each network is then diffused
    through the *others'* local structure, repeatedly, until they agree. An edge
    that several blocks support is reinforced; an edge only one block reports
    decays. That is a per-protein operation, which is what `graph` has over
    `late`: `late` applies one weight to every protein, while here a protein
    whose blocks agree comes out sharp and one whose blocks disagree comes out
    diffuse, without anyone configuring either.

    The per-protein mixture is recovered afterwards and recorded on each
    contribution, which is where FOLLOWUPS #16's missing per-protein weights
    actually live -- on the output, not in the config.

    Needs at least two blocks: the update step diffuses each network through the
    average of the *others*, and with one network there are no others.

    **What is the paper's and what is this implementation's.** The kernel (eq.
    3), the full normalization (eq. 8), the local k-nearest normalization (eq.
    9) and the cross-diffusion update are the paper's. Two things are not:
    weighting the cross-view average by the configured block weights, which the
    paper has no notion of; and re-applying the full normalization after every
    update rather than only at the start, which keeps each iterate
    row-stochastic so that the per-step delta is a comparable quantity and the
    convergence warning below means something. No reference implementation was
    available in this environment to check the output against -- `snfpy` is not
    installed and ADR 0006 keeps it that way -- so the tests check the
    algorithm's *properties* (symmetry, row sums, convergence, that two
    identical views split every protein exactly evenly, that an informative view
    beats a noise view per protein, and that raising a weight moves the result
    toward that block) rather than agreement with another implementation.

    **The emitted profile has a zeroed diagonal, and that is load-bearing.**
    Equation 8 pins every protein's self-affinity at exactly 1/2 while its
    affinities to everything else are O(1/N) -- here, 0.5 against 0.001. Left in,
    that one entry dominates the euclidean distance between any two rows, so
    *every* pair of proteins sits about sqrt(2)/2 apart and the profile carries
    no structure at all: measured on the fixture, the fused map's separation of
    a planted partition went from 1.00 (invisible) to 1.39 (recovered) purely by
    zeroing it. The self-affinity is a normalization constant rather than
    evidence -- it is the same 1/2 for every protein and says nothing about who
    anyone is near -- so removing it discards no information. The same reasoning
    applies to the per-protein shares below, which are computed off-diagonal for
    the same reason and were likewise uniform at 0.500/0.499 before.
    """
    inputs = _check_inputs(inputs, "graph", minimum=2)
    n = inputs[0].n_proteins
    if not 1 <= k <= n:
        raise FusionError(
            f"graph fusion: k must be between 1 and N ({n}), got {k}. k counts the "
            "protein itself, so k=1 keeps only the diagonal and fuses nothing."
        )
    if not 0 < mu:
        raise FusionError(f"graph fusion: mu must be positive, got {mu}.")
    if iterations < 1:
        raise FusionError(f"graph fusion: iterations must be at least 1, got {iterations}.")

    normalized, raw_means = _normalized_distances(inputs)
    weights = np.array([block.weight for block in inputs], dtype=np.float64)

    initial = []
    local = []
    for distances in normalized:
        affinity = _scaled_exponential_kernel(distances, k, mu)
        initial.append(_full_normalize(affinity))
        local.append(_local_normalize(affinity, k))

    status = list(initial)
    deltas = []
    for _ in range(iterations):
        updated = []
        for v in range(len(status)):
            others = weights.copy()
            others[v] = 0.0
            if others.sum() <= 0:
                # Every other block has weight zero, so this one diffuses
                # through nothing and is left as it stands.
                updated.append(status[v])
                continue
            average = sum(w * p for w, p in zip(others, status)) / others.sum()
            propagated = local[v] @ average @ local[v].T
            updated.append(_full_normalize(propagated))
        deltas.append(float(max(np.abs(a - b).max() for a, b in zip(updated, status))))
        status = updated

    fused = sum(w * p for w, p in zip(weights, status)) / weights.sum()
    fused = 0.5 * (fused + fused.T)
    # Drop the self-affinity, which eq. 8 pins at 1/2 for every protein. See the
    # docstring: it is a normalization constant, it is identical for everyone,
    # and left in place it is the only thing any two rows differ by.
    np.fill_diagonal(fused, 0.0)

    # Per-protein mixture: how much of each protein's fused neighborhood is
    # explained by each block's *original* evidence. The initial affinities are
    # used rather than the diffused ones on purpose -- after diffusion every
    # network has been pulled toward the others, so scoring against them would
    # measure how well the diffusion converged rather than which block spoke.
    # Off-diagonal for the same reason as above: with the 1/2 left in, every
    # inner product is dominated by a term both blocks share exactly.
    initial_off = []
    for p in initial:
        without_self = p.copy()
        np.fill_diagonal(without_self, 0.0)
        initial_off.append(without_self)
    agreement = np.stack(
        [w * np.einsum("ij,ij->i", fused, p) for w, p in zip(weights, initial_off)]
    )
    totals = agreement.sum(axis=0)
    uniform = np.full(len(inputs), 1.0 / len(inputs))
    per_protein = np.where(
        totals > 0, agreement / np.where(totals > 0, totals, 1.0), uniform[:, None]
    )
    global_share = per_protein.mean(axis=1)
    global_share = global_share / global_share.sum()

    contributions = tuple(
        BlockContribution(
            block_id=block.block_id,
            weight=block.weight,
            n_features=block.n_features,
            share=float(weights[i] / weights.sum()),
            realized_share=float(global_share[i]),
            mean_distance=raw_means[i],
            dispersion=float(_off_diagonal(normalized[i]).var()),
            per_protein_share=per_protein[i],
        )
        for i, block in enumerate(inputs)
    )

    warnings = list(_dominance_warning(contributions))
    if deltas and deltas[-1] > 1e-6:
        warnings.append(
            f"graph fusion had not converged after {iterations} iteration(s): the "
            f"last step still moved an affinity by {deltas[-1]:.3g}. Raise "
            "`iterations`, or read the fused map as provisional."
        )

    return FusionResult(
        values=fused,
        strategy="graph",
        representation="affinity_profile",
        contributions=contributions,
        warnings=tuple(warnings),
        params_used={
            "strategy": "graph",
            "algorithm": "SNF (Wang et al. 2014)",
            "k": int(k),
            "mu": float(mu),
            "iterations": int(iterations),
            "normalization": "unit_mean_distance",
            "final_delta": deltas[-1] if deltas else None,
        },
    )


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

#: Which `params` keys each strategy accepts. Anything else is rejected rather
#: than ignored: a misspelled `iteratons` that is silently dropped leaves a run
#: that reports the default and looks configured.
STRATEGY_PARAMS = {
    "none": frozenset(),
    "early": frozenset(),
    "late": frozenset(),
    "graph": frozenset({"k", "mu", "iterations"}),
}


def fuse(strategy: str, inputs, params: dict | None = None) -> FusionResult:
    """Run one strategy. The only entry point callers need."""
    if strategy not in FUSION_STRATEGIES:
        raise FusionError(
            f"unknown fusion strategy {strategy!r}. Allowed: "
            f"{', '.join(sorted(FUSION_STRATEGIES))}."
        )
    params = dict(params or {})
    unknown = sorted(set(params) - STRATEGY_PARAMS[strategy])
    if unknown:
        allowed = sorted(STRATEGY_PARAMS[strategy])
        raise FusionError(
            f"strategy {strategy!r} does not take parameter(s) {unknown}. "
            f"Allowed: {allowed or '(none)'}."
        )
    if strategy == "none":
        return fuse_none(inputs)
    if strategy == "early":
        return fuse_early(inputs)
    if strategy == "late":
        return fuse_late(inputs)
    return fuse_graph(inputs, **params)
