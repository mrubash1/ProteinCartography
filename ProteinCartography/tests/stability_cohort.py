#!/usr/bin/env python
"""Proteins whose neighborhoods are stable, fragile, or pure chance, by design.

Neighborhood stability asks whether a protein's nearest neighbors are a finding
or a coin flip. Its failure mode is not a crash and not a wrong magnitude -- it
is reporting *confidence in an arbitrary answer*. A protein sitting among forty
exactly equidistant candidates has a perfectly deterministic k-nearest list,
because `argsort` breaks ties by index, and a diagnostic that resamples nothing
and jitters nothing will call that list stable. It is not stable; it is
arbitrary and reproducible, which is the worst combination a map can offer.

Group 8a met the same shape from another direction: `graph` fusion returned a
matrix that was symmetric, row-stochastic, finite and correctly shaped, and
carried no structure at all. Shape assertions cannot see this.
Only a planted answer the statistic is *required* to reproduce can.

So this plants three bands, and the middle one is as load-bearing as the ends:

===========  ==============================================  =================
group        construction                                    should score
===========  ==============================================  =================
`gapped`     clusters of exactly ``k + 1``, tiny radius,      **high** -- the
             centers drawn from the same cloud as            k nearest are
             everything else                                 forced
`diffuse`    isotropic cloud at the ambient scale            **middling** --
                                                             real but fragile
`tied`       mutually equidistant points (``a * e_j``),       **chance** -- and
             clustered away from the cloud                   provably so
===========  ==============================================  =================

**All three sit at one distance scale on purpose.** An earlier draft separated
the groups by 100x so they could not interfere, and it inverted the answer: the
noise level is scaled by the matrix's median pairwise distance, that median
became a *between-group* distance, and the resulting sigma was twenty times the
`gapped` clusters' radius. The tight, well-separated group scored 0.298 and the
structureless one scored 0.443. Nothing was wrong with the statistic. The
fixture had confounded local structure with global scale, and it read as the
diagnostic being backwards.

What separates the bands here is therefore purely **local**: the size of the gap
between a protein's k-th and (k+1)-th neighbor, relative to a noise level every
protein shares.

Measured at the defaults (N=166, k=10, 20 replicates, 80% subsample), which
``tests/test_stability_cohort.py`` pins:

============  ========  ========  ========  ========
group          noise 0   n. 0.01   n. 0.05   n. 0.10
============  ========  ========  ========  ========
`gapped`        1.0000    0.9583    0.9196    0.8724
`diffuse`       1.0000    0.7824    0.6228    0.4953
`tied`          1.0000    0.1968    0.1968    0.1968
============  ========  ========  ========  ========

At the module's default noise level of 0.10 each group lands in a different
reporting band with margins of 0.172, 0.195 and 0.103 -- one group per band, and
the bands are only a test if the margins are stated.

Three properties of that table are the fixture, and each is asserted:

**Zero noise scores exactly 1.0, for every protein.** Not approximately. The
reference neighbor set is recomputed *within each subsample* rather than taken
from the full cohort, so resampling alone cannot change the answer -- both sides
lose the same proteins and both promote the same replacement. That makes the
subsample a genuine null: any departure from 1.0 is attributable to the noise
term and to nothing else. It is also the negative control for the harness, in
the sense group 8b's determinism guard is one: a perturbation that perturbs
nothing must score perfectly, and a statistic that cannot return exactly 1.0
here is measuring its own bookkeeping.

**`tied` is flat in the noise level, and its value is predicted in closed
form.** Any sigma above zero randomizes an exact tie completely, so the score
does not decay with noise -- it steps straight to chance and stays there. That
chance level is :func:`chance_jaccard`: two independent uniform ``k``-subsets of
a pool of ``p`` candidates overlap in ``k^2/p`` elements on average, giving a
Jaccard of ``k^2/p / (2k - k^2/p)``. At these defaults it predicts 0.1923 and
the measurement is 0.1968. A diagnostic that reported anything
appreciably *above* chance for this group would be reading the tie-break order
of `argsort` and calling it structure.

**The ordering survives the seed.** `gapped` > `diffuse` > `tied` holds at every
seed in ``range(6)``, and so does one-group-per-band against the module's two
thresholds. The bands are only a test if they are not a lucky draw.

Reproducible from ``(k, seed)`` alone; writes nothing and holds no state.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from spaces.base import BlockResult, BlockSpec

__all__ = [
    "AMBIENT_DIMENSIONS",
    "DEFAULT_K",
    "DEFAULT_SEED",
    "DIFFUSE",
    "DIFFUSE_N",
    "GAPPED",
    "GAPPED_CLUSTERS",
    "GAPPED_RADIUS",
    "MINIMUM_CLUSTER_SEPARATION",
    "TIED",
    "TIED_CENTRE_DISTANCE",
    "TIED_DISTANCE",
    "TIED_N",
    "StabilityCohort",
    "chance_jaccard",
    "stability_cohort",
]

DEFAULT_SEED = 0

#: Phase 5 item 3 specifies k=10 for neighborhood stability. The pipeline's
#: `diagnostics.k` defaults to 15 and the entry point passes that; this is the
#: fixture's own k, and `gapped`'s cluster size is tied to it, so the two move
#: together rather than one silently outgrowing the other.
DEFAULT_K = 10

GAPPED = "gapped"
DIFFUSE = "diffuse"
TIED = "tied"

AMBIENT_DIMENSIONS = 8
GAPPED_CLUSTERS = 6
DIFFUSE_N = 60
TIED_N = 40

#: Small enough that a `gapped` cluster's whole membership is nearer to each of
#: its members than any other protein is, by a wide margin. The margin is the
#: point: it is what the noise term has to overcome, and it is checked directly
#: by ``test_the_gapped_clusters_have_a_real_gap_at_k``.
#:
#: What binds it is not the spacing between clusters -- that is imposed by
#: :data:`MINIMUM_CLUSTER_SEPARATION` -- but the nearest `diffuse` point, which
#: is drawn from the same cloud the centers are and can land anywhere. At 0.04
#: the closest such interloper left a rank-k gap ratio of 5.9. Shrinking the
#: blob is the fix that costs nothing: Jaccard scores *sets*, and a tighter
#: cluster does not make the set any less forced.
GAPPED_RADIUS = 0.01

#: `gapped` cluster centers are rejection-sampled to be at least this far
#: apart. Drawn i.i.d. from the cloud they are usually well separated and
#: occasionally not: at seed 0 the closest pair left a rank-k gap ratio of 5.9
#: where the others were above 8.6. The band still held, but a margin that
#: depends on the draw is not a construction -- so it is imposed.
MINIMUM_CLUSTER_SEPARATION = 2.0

#: Every pair of `tied` proteins is exactly this far apart. Chosen below the
#: ambient cloud's typical spacing so that a tied protein's neighbors are other
#: tied proteins -- otherwise the group is a mixture and its chance level is not
#: the closed form.
TIED_DISTANCE = 1.0

#: How far the `tied` group sits from the cloud's center. Large enough that no
#: tied protein has a non-tied neighbor, small enough that the group does not
#: dominate the median pairwise distance the noise level is scaled by.
TIED_CENTRE_DISTANCE = 3.0


@dataclass(frozen=True)
class StabilityCohort:
    """One feature matrix over one protein index, with a per-protein truth."""

    protids: list
    features: np.ndarray
    #: One group name per protein, in ``protids`` order.
    groups: np.ndarray
    k: int

    @property
    def n_proteins(self) -> int:
        return len(self.protids)

    def members(self, group: str) -> np.ndarray:
        """Positions of one group's proteins."""
        found = np.flatnonzero(self.groups == group)
        if found.size == 0:
            raise ValueError(f"no group named {group!r}; have {sorted(set(self.groups))}")
        return found

    def protids_in(self, group: str) -> list:
        return [self.protids[i] for i in self.members(group)]

    def distances(self) -> np.ndarray:
        """Euclidean pairwise distances, by explicit broadcasting.

        The fixture's own measuring stick, computed the slow obvious way rather
        than through ``coregistration.pairwise_distances``. A measuring stick
        that shares an implementation with the thing it measures cannot catch
        that implementation being wrong -- the same reason
        ``fusion_cohort.separation`` avoids the Gram identity.
        """
        values = np.asarray(self.features, dtype=np.float64)
        return np.sqrt(((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2))

    def block_spec(self, block_id: str = "stability") -> BlockSpec:
        return BlockSpec(
            id=block_id,
            kind="features",
            fusable=True,
            metric="euclidean",
            normalization="none",
            provider="stability_cohort",
        )

    def block_result(self, block_id: str = "stability") -> BlockResult:
        """A real :class:`BlockResult`, so tests can drive the real path."""
        return BlockResult(
            spec=self.block_spec(block_id),
            protids=list(self.protids),
            features=np.asarray(self.features, dtype=np.float64),
        )


def chance_jaccard(pool: int, k: int) -> float:
    """Expected Jaccard of two independent uniform ``k``-subsets of ``pool``.

    The level a neighborhood carrying no information scores. Two such subsets
    share ``k^2 / pool`` elements in expectation, and the union of two
    ``k``-sets sharing ``m`` is ``2k - m``, so the ratio follows. It is the
    expectation of a ratio approximated by the ratio of expectations, which is
    why ``tests/test_stability_cohort.py`` allows 0.02 against the measurement
    rather than asserting equality.

    Args:
        pool: candidates available, excluding the protein itself.
        k: neighborhood size.
    """
    if pool < k or k < 1:
        raise ValueError(f"need 1 <= k <= pool, got k={k}, pool={pool}")
    shared = k * k / pool
    return float(shared / (2 * k - shared))


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _separated_centres(rng: np.random.RandomState) -> np.ndarray:
    """``GAPPED_CLUSTERS`` cloud points at least ``MINIMUM_CLUSTER_SEPARATION``
    apart.

    Rejection sampling rather than a lattice: the centers must look like
    ordinary draws from the same cloud the `diffuse` group comes from, because
    the fixture's claim is that only *local* structure separates the bands. A
    lattice would give the `gapped` group a global regularity `diffuse` lacks
    and hand the diagnostic a second signal to read.
    """
    centres = []
    for _ in range(10000):
        candidate = rng.normal(0.0, 1.0, size=AMBIENT_DIMENSIONS)
        if all(np.linalg.norm(candidate - c) >= MINIMUM_CLUSTER_SEPARATION for c in centres):
            centres.append(candidate)
            if len(centres) == GAPPED_CLUSTERS:
                return np.array(centres)
    raise RuntimeError(
        f"could not place {GAPPED_CLUSTERS} centers {MINIMUM_CLUSTER_SEPARATION} apart "
        f"in {AMBIENT_DIMENSIONS} dimensions; loosen the separation or widen the cloud."
    )


def stability_cohort(*, k: int = DEFAULT_K, seed: int = DEFAULT_SEED) -> StabilityCohort:
    """Generate the cohort. Deterministic in ``(k, seed)`` alone."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    rng = np.random.RandomState(seed)
    tied_leg = TIED_DISTANCE / np.sqrt(2.0)

    blocks, groups = [], []

    # `gapped`: clusters of exactly k + 1, so the (k+1)-th neighbor of any
    # member is in a *different* cluster. That is what puts the gap at rank k
    # rather than somewhere past it -- a tight blob larger than k + 1 has its
    # members mutually equidistant at rank k and behaves like `tied`.
    for centre in _separated_centres(rng):
        blob = centre + rng.normal(0.0, GAPPED_RADIUS, size=(k + 1, AMBIENT_DIMENSIONS))
        blocks.append(np.hstack([blob, np.zeros((k + 1, TIED_N))]))
        groups += [GAPPED] * (k + 1)

    diffuse = rng.normal(0.0, 1.0, size=(DIFFUSE_N, AMBIENT_DIMENSIONS))
    blocks.append(np.hstack([diffuse, np.zeros((DIFFUSE_N, TIED_N))]))
    groups += [DIFFUSE] * DIFFUSE_N

    # `tied`: a scaled identity gives every pair the same separation,
    # a * sqrt(2), in exact arithmetic -- no near-ties, no tolerance.
    centre = _unit(rng.normal(0.0, 1.0, size=AMBIENT_DIMENSIONS)) * TIED_CENTRE_DISTANCE
    tied = np.hstack([np.tile(centre, (TIED_N, 1)), np.eye(TIED_N) * tied_leg])
    blocks.append(tied)
    groups += [TIED] * TIED_N

    features = np.vstack(blocks)
    protids = [f"SC{i:04d}" for i in range(features.shape[0])]
    return StabilityCohort(
        protids=protids,
        features=features,
        groups=np.array(groups),
        k=k,
    )
