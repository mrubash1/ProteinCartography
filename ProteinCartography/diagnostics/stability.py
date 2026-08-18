#!/usr/bin/env python
"""Is this protein's neighbor list a finding, or a coin flip?

Phase 5 item 3. Every other diagnostic in this package asks whether a map is
faithful to its input. This one asks whether the *input* was ever determinate:
a protein whose k-th and (k+1)-th neighbors are separated by less than the
measurement's own noise has a neighbor list that would have come out differently
had the run been repeated, and no amount of faithful projection makes that list
mean anything.

The statistic is per-protein Jaccard overlap between the k-nearest set in the
data and the k-nearest set under a perturbation, averaged over replicates. Two
perturbations are applied together, both of them modelling something the
pipeline actually does:

**Resample the retrieved cohort.** The proteins in a ProteinCartography run are
a *retrieval result*, not a fixed population -- a different Foldseek call
returns a different set. Each replicate draws ``subsample_fraction`` of them
without replacement. Without replacement, not bootstrap-with-replacement:
duplicated proteins sit at distance zero from each other and would occupy the
whole neighborhood.

**Add score noise.** Gaussian, symmetric, scaled to a fraction of the matrix's
median pairwise distance so that it means the same thing on a TM-score block and
on a physicochemistry block, whose scales differ by orders of magnitude (ADR
0002's problem, in a different place).

**The reference is recomputed inside each subsample**, not taken from the full
cohort. That makes resampling a genuine null: both sides lose the same proteins
and both promote the same replacement, so a replicate with no noise scores
exactly 1.0 and every departure is attributable to the noise term alone. It also
means ``subsample_fraction`` is not decoration -- a smaller cohort has fewer
candidates competing for the k-th slot, so the same noise flips fewer of them.

Two failure modes this is built against, both found in earlier groups:

* **Reporting confidence in an arbitrary answer.** Exact ties are common here --
  censored TM-scores arrive as thousands of exact zeros -- and ``argsort``
  resolves them by index, deterministically. A diagnostic that perturbs nothing
  calls that stable. ``tests/stability_cohort`` plants forty mutually
  equidistant proteins whose score is predicted in closed form by chance, and
  the statistic is required to land there.
* **A default that cannot fit the cohort.** Group 8b shipped ``DEFAULT_K = 15``
  against an 11-protein demo and broke all seven spaces while every unit test
  passed at N=240 (REVIEW_LOG G8b.5). Here the ceiling is tighter still, because
  k must fit the *subsample* rather than the cohort, so it is clamped to
  ``round(f*N) - 1`` and the request is kept and reported.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from diagnostics.embedding import neighbor_ordering

__all__ = [
    "COIN_FLIP_THRESHOLD",
    "DEFAULT_NOISE",
    "DEFAULT_REPLICATES",
    "DEFAULT_SUBSAMPLE_FRACTION",
    "STABLE_THRESHOLD",
    "VACUOUS_FRACTION",
    "NeighborhoodStability",
    "StabilityError",
    "jaccard_rows",
    "largest_stable_k",
    "neighborhood_stability",
]

DEFAULT_REPLICATES = 20
DEFAULT_SUBSAMPLE_FRACTION = 0.8

#: A tenth of the median pairwise distance. Deliberately a stress level rather
#: than an estimate of any instrument's error, which nothing here could know:
#: the question the diagnostic answers is "would this neighborhood survive a
#: perturbation of this size", and a perturbation too small to move anything
#: makes every protein look determinate. Measured on
#: ``tests/stability_cohort`` this is the level at which the three planted
#: bands land one per reporting band, with margins of 0.172, 0.195 and 0.103.
DEFAULT_NOISE = 0.10

#: Reporting bands, on the same footing as ``embedding``'s: descriptions of
#: where a line was drawn, not thresholds from any literature. Phase 5 states
#: the lower one -- "a point at 0.3 stability is a coin flip, not a finding".
STABLE_THRESHOLD = 0.70
COIN_FLIP_THRESHOLD = 0.30

#: Above this share of the subsample, a "neighborhood" is most of the cohort
#: and the score stops discriminating. Found by running the demo rather than by
#: reasoning: at eleven proteins k clamps to 8 and a subsample holds 9, so every
#: protein's k nearest are *all* the others and the Jaccard is 1.0 whatever the
#: noise. All seven demo spaces reported perfect stability under a sigma half
#: the size of the data. That is not a stable cohort; it is a statistic with no
#: room left to be wrong in, and reporting 1.000 for it is the most confident
#: possible way to say nothing.
VACUOUS_FRACTION = 0.5


class StabilityError(ValueError):
    """Raised when the cohort cannot support the requested measurement."""


def largest_stable_k(n: int, subsample_fraction: float) -> int:
    """The largest k a subsample of this cohort can supply neighbors for.

    ``round(f * n) - 1``: a replicate holds that many proteins and one of them
    is the protein being scored. Returns 0 when no replicate can supply even a
    single neighbor, which the caller reports rather than dividing by.
    """
    return max(0, int(round(subsample_fraction * n)) - 1)


def jaccard_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Row-wise Jaccard of two ``(m, k)`` arrays of neighbor indices.

    Both rows hold exactly k distinct entries, so ``|A u B| = 2k - |A n B|``
    and only the intersection has to be counted. Done by sorting each pair of
    rows together and counting adjacent equal values, which is O(k log k) and
    avoids building a Python set per protein per replicate -- at 20 replicates
    over a few thousand proteins that inner loop is the whole runtime.
    """
    if left.shape != right.shape:
        raise StabilityError(
            f"neighbor sets must be the same shape, got {left.shape}, {right.shape}"
        )
    k = left.shape[1]
    if k == 0:
        raise StabilityError("cannot take the Jaccard of empty neighbor sets")
    together = np.sort(np.concatenate([left, right], axis=1), axis=1)
    shared = (together[:, 1:] == together[:, :-1]).sum(axis=1)
    return shared / (2 * k - shared)


def _median_offdiagonal(distances: np.ndarray) -> float:
    n = distances.shape[0]
    return float(np.median(distances[~np.eye(n, dtype=bool)]))


def _symmetric_noise(size: int, sigma: float, rng) -> np.ndarray:
    """Gaussian noise that keeps the matrix symmetric and hollow.

    Symmetrized rather than drawn once per unordered pair: the two are the same
    distribution up to a factor of sqrt(2) in the variance, and this way the
    draw count depends only on ``size``, which keeps a replicate's stream
    position independent of anything measured from the data.
    """
    error = rng.normal(0.0, sigma, size=(size, size))
    error = (error + error.T) / 2.0
    np.fill_diagonal(error, 0.0)
    return error


@dataclass(frozen=True)
class NeighborhoodStability:
    """One space's per-protein stability, and what was done to measure it."""

    space_id: str
    k: int
    protids: list
    #: Mean Jaccard per protein, in ``[0, 1]``. NaN for a protein that no
    #: replicate happened to retain, which is possible at small N and is
    #: reported rather than silently averaged over.
    stability: np.ndarray
    #: How many replicates each protein appeared in. An estimate from three
    #: replicates is not the same evidence as one from twenty, and the
    #: difference is invisible in the mean.
    replicates_seen: np.ndarray
    replicates: int
    subsample_fraction: float
    #: How many proteins each replicate actually held. The denominator that
    #: matters for whether k leaves any room, and not recoverable from the
    #: fraction alone once rounding is involved.
    subsample_size: int
    noise: float
    noise_sigma: float
    #: The k that was asked for, when it did not fit the subsample.
    k_requested: int = 0

    @property
    def mean_stability(self) -> float:
        return float(np.nanmean(self.stability))

    @property
    def measured(self) -> np.ndarray:
        return ~np.isnan(self.stability)

    @property
    def neighborhood_fraction(self) -> float:
        """k as a share of the candidates a replicate offers."""
        return self.k / max(1, self.subsample_size - 1)

    @property
    def informative(self) -> bool:
        """False when k leaves the statistic no room to discriminate."""
        return self.neighborhood_fraction < VACUOUS_FRACTION

    def coin_flips(self) -> list:
        """Proteins at or below :data:`COIN_FLIP_THRESHOLD`."""
        return [
            self.protids[i]
            for i in np.flatnonzero(self.measured & (self.stability <= COIN_FLIP_THRESHOLD))
        ]

    def stable(self) -> list:
        return [
            self.protids[i]
            for i in np.flatnonzero(self.measured & (self.stability >= STABLE_THRESHOLD))
        ]

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(
            {"stability": self.stability, "replicates_seen": self.replicates_seen},
            index=pd.Index(self.protids, name="protid"),
        )

    def to_dict(self) -> dict:
        coin_flips = self.coin_flips()
        return {
            "space_id": self.space_id,
            "k": self.k,
            "k_requested": self.k_requested,
            "replicates": self.replicates,
            "subsample_fraction": self.subsample_fraction,
            "noise": self.noise,
            "noise_sigma": self.noise_sigma,
            "n_proteins": len(self.protids),
            "n_measured": int(self.measured.sum()),
            "subsample_size": self.subsample_size,
            "neighborhood_fraction": self.neighborhood_fraction,
            "informative": self.informative,
            "stability_mean": self.mean_stability,
            "stability_min": float(np.nanmin(self.stability)),
            "n_stable": len(self.stable()),
            "n_coin_flips": len(coin_flips),
            "coin_flips": coin_flips,
            "warnings": self.warnings(),
        }

    def warnings(self) -> list:
        notes = []
        if self.k_requested and self.k_requested != self.k:
            notes.append(
                f"k was reduced from {self.k_requested} to {self.k}: a "
                f"{self.subsample_fraction:.0%} subsample of {len(self.protids)} proteins "
                f"leaves {int(round(self.subsample_fraction * len(self.protids)))}, which "
                f"can supply at most {self.k} neighbors."
            )
        if self.k >= self.subsample_size - 1:
            notes.append(
                f"k={self.k} and a replicate holds {self.subsample_size} proteins, so every "
                "protein's k nearest are all the others and the Jaccard is 1.0 by "
                "construction. This section measures nothing on a cohort this small."
            )
        elif not self.informative:
            notes.append(
                f"k={self.k} is {self.neighborhood_fraction:.0%} of the "
                f"{self.subsample_size - 1} candidates a replicate offers, so a "
                '"neighborhood" here is most of the cohort. The score is real but it '
                "is not measuring anything local."
            )
        unmeasured = int((~self.measured).sum())
        if unmeasured:
            notes.append(
                f"{unmeasured} of {len(self.protids)} proteins appeared in no replicate and "
                "have no stability value. Raise `diagnostics.bootstrap_replicates`."
            )
        flips = len(self.coin_flips())
        if flips:
            notes.append(
                f"{flips} of {int(self.measured.sum())} measured proteins have neighborhoods "
                f"at or below {COIN_FLIP_THRESHOLD:.2f} under {self.noise:.0%} score noise -- "
                "their neighbor lists are close to a coin flip and should not be read as "
                "findings."
            )
        if self.mean_stability <= COIN_FLIP_THRESHOLD:
            notes.append(
                f"mean stability is {self.mean_stability:.3f}, at or below the coin-flip "
                "level. Either this space has no reliable neighborhood structure, or its "
                "distances are dominated by exact ties -- a censored similarity matrix read "
                "without its mask looks exactly like this."
            )
        return notes


def neighborhood_stability(
    space_id: str,
    distances: np.ndarray,
    protids: list,
    *,
    k: int,
    replicates: int = DEFAULT_REPLICATES,
    subsample_fraction: float = DEFAULT_SUBSAMPLE_FRACTION,
    noise: float = DEFAULT_NOISE,
    seed: int = 0,
) -> NeighborhoodStability:
    """Per-protein kNN Jaccard over resampled, noised replicates.

    Args:
        space_id: for the report only.
        distances: square pairwise distances over ``protids``, in that order.
        protids: the space's proteins.
        k: neighborhood size, clamped down to what a subsample can supply.
        replicates: how many perturbed draws to average over.
        subsample_fraction: proportion of the cohort each replicate retains.
        noise: Gaussian sigma as a fraction of the median pairwise distance.
        seed: the whole result is a deterministic function of this and the data.
    """
    distances = np.asarray(distances, dtype=np.float64)
    n = len(protids)
    if distances.ndim != 2 or distances.shape != (n, n):
        raise StabilityError(
            f"distances must be square over the {n} protids, got shape {distances.shape}"
        )
    if not 0 < subsample_fraction <= 1:
        raise StabilityError(f"subsample_fraction must be in (0, 1], got {subsample_fraction}")
    if noise < 0:
        raise StabilityError(f"noise must not be negative, got {noise}")
    if replicates < 1:
        raise StabilityError(f"replicates must be at least 1, got {replicates}")
    if k < 1:
        raise StabilityError(f"k must be at least 1, got {k}")

    ceiling = largest_stable_k(n, subsample_fraction)
    if ceiling < 1:
        raise StabilityError(
            f"a {subsample_fraction:.0%} subsample of {n} proteins leaves "
            f"{int(round(subsample_fraction * n))}, which cannot supply a single neighbor. "
            "Raise `diagnostics.subsample_fraction`, or do not ask for stability on a "
            "cohort this small."
        )
    used = min(k, ceiling)

    sigma = noise * _median_offdiagonal(distances)
    rng = np.random.RandomState(seed)
    size = int(round(subsample_fraction * n))
    totals = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)

    for _ in range(replicates):
        # Sorted so the submatrix keeps the cohort's own protid order, which
        # makes the argsort tie-break inside a replicate the same one the full
        # matrix would use. Without it the reference and the replicate would
        # still agree -- they share the draw -- but two replicates covering the
        # same proteins would not, and the noise-free case would stop being
        # exactly 1.0 for reasons that have nothing to do with stability.
        drawn = np.sort(rng.choice(n, size=size, replace=False))
        block = distances[np.ix_(drawn, drawn)]
        reference = neighbor_ordering(block)[:, 1 : used + 1]
        perturbed = block + _symmetric_noise(size, sigma, rng)
        np.fill_diagonal(perturbed, 0.0)
        replicate = neighbor_ordering(perturbed)[:, 1 : used + 1]
        totals[drawn] += jaccard_rows(reference, replicate)
        counts[drawn] += 1

    stability = np.full(n, np.nan)
    seen = counts > 0
    stability[seen] = totals[seen] / counts[seen]
    return NeighborhoodStability(
        space_id=space_id,
        k=used,
        k_requested=k,
        protids=list(protids),
        stability=stability,
        replicates_seen=counts,
        replicates=replicates,
        subsample_fraction=float(subsample_fraction),
        subsample_size=size,
        noise=float(noise),
        noise_sigma=float(sigma),
    )
