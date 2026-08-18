#!/usr/bin/env python
"""Cluster one space, using the clustering the pipeline already ships.

Phase 5 items 7 and 8 are diagnostics *about* a partition, and until now no
space had one: ``reduce_space`` emits coordinates, and the pipeline's only
clustering is Leiden over the TM-score matrix in the legacy path. This supplies
the missing capability, and the way it supplies it is the decision worth
recording (ADR 0015).

**It is scanpy's Leiden, not a reimplementation.** Everything else numeric in
this work is hand-written numpy -- the three co-registration metrics, four
enrichment statistics, four fusion strategies including SNF, trustworthiness
and continuity. Leiden is not in that class, and neither is the reason for
hand-writing them. Three things decide it:

* **It adds no dependency.** ``scanpy=1.9.3`` and ``leidenalg=0.9.1`` are
  already pinned in ``envs/analysis.yml`` and already run in the default DAG,
  in ``rule leiden_clustering``. ADR 0006 governs *optional and licensed*
  dependencies; this is neither, and the rule that consumes this module
  declares that same environment.
* **A diagnostic about a partition must be about the partition that ships.** A
  hand-rolled clusterer would make item 7 sweep the resolution of an algorithm
  that never produces ``leiden_features.tsv``, and item 8 a control for a
  method nobody runs. Being *approximately* the pipeline's clustering is worse
  here than not clustering at all, because the numbers would look comparable
  and not be.
* **Leiden is not SNF.** Group 8a hand-rolled similarity network fusion in
  sixty lines because it has a closed form. Leiden has a local-moving phase, a
  refinement phase and an aggregation phase, and its refinement guarantee --
  that every returned community is internally well connected -- is the whole
  reason it is preferred over Louvain. An approximation carrying the name would
  be worse than no implementation.

The cost is that the reference implementation *is* the implementation, so there
is nothing to cross-check the arithmetic against. What replaces that is
``tests/test_clustering.py``'s agreement test: this module and
``leiden_clustering.scanpy_leiden_cluster`` are required to return the same
partition of the same matrix, so the per-space path cannot drift away from the
legacy one without a test failing. That is the same argument
``spaces/reducers/core.py`` makes for sharing the PCA, arrived at from the other
direction -- shared code there, an enforced equality here, because
``leiden_clustering.py`` is a pre-existing file and this work does not refactor
those without cause.

The import is deferred and :func:`is_available` reports on it, so that
importing this module costs nothing in an environment without scanpy and the
unit suite can exercise everything around it.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_N_NEIGHBORS",
    "DEFAULT_N_PCS",
    "DEFAULT_RESOLUTION",
    "DEFAULT_SEED",
    "ClusteringError",
    "Partition",
    "is_available",
    "leiden_partition",
    "sweep_resolutions",
]

#: Both match ``leiden_clustering.py``'s argparse defaults, which are what the
#: Snakefile passes today. A per-space partition computed at other settings
#: would not be comparable to the legacy one, and the agreement test would be
#: comparing two different questions.
DEFAULT_N_NEIGHBORS = 10
DEFAULT_N_PCS = 30

DEFAULT_RESOLUTION = 1.0

#: scanpy's own default. Named here because Leiden is stochastic and a
#: diagnostic that changed between two runs of the same data would be
#: indistinguishable from the instability it is trying to measure -- which is
#: the defect ``tests/test_determinism.py`` exists to catch in the reducers.
DEFAULT_SEED = 0


class ClusteringError(RuntimeError):
    """Raised when this space cannot be clustered at all."""


def is_available() -> tuple:
    """``(available, explanation)``, in the shape ADR 0006 rule 2 specifies."""
    try:
        import scanpy  # noqa: F401
    except ImportError as error:
        return False, (
            f"scanpy is not importable ({error}). It is pinned in envs/analysis.yml, "
            "which is the environment `rule diagnose_space` declares; this path is "
            "only reached outside that environment."
        )
    try:
        import leidenalg  # noqa: F401
    except ImportError as error:
        return False, (
            f"scanpy is present but leidenalg is not ({error}), so sc.tl.leiden cannot "
            "run. Both are pinned in envs/analysis.yml."
        )
    return True, "scanpy and leidenalg are importable"


@dataclass(frozen=True)
class Partition:
    """One clustering of one space, and the settings that produced it."""

    protids: list
    #: One label per protein, in ``protids`` order.
    labels: list
    resolution: float
    n_neighbors: int
    n_pcs: int
    seed: int

    @property
    def n_clusters(self) -> int:
        return len(set(self.labels))

    def as_mapping(self) -> dict:
        return dict(zip(self.protids, self.labels))

    def to_frame(self, cluster_name: str = "cluster"):
        import pandas as pd

        return pd.DataFrame(
            {cluster_name: self.labels}, index=pd.Index(self.protids, name="protid")
        )


def _clamped(n: int, n_vars: int, n_neighbors: int, n_pcs: int) -> tuple:
    """The clamping ``leiden_clustering.scanpy_leiden_cluster`` performs.

    Reproduced rather than imported because importing it would pull scanpy in
    at module scope. The agreement test is what holds the duplication honest:
    if these drift, the two paths return different partitions and it fails.
    """
    recommended = int(np.round(n / 10))
    used_neighbors = max(n_neighbors, recommended)
    # scanpy requires 1 < n_neighbors <= N - 1 for a connected graph.
    used_neighbors = max(2, min(used_neighbors, n - 1))
    max_pcs = max(1, min(n - 1, n_vars - 1 if n_vars > 1 else 1))
    return used_neighbors, max(1, min(n_pcs, max_pcs))


def leiden_partition(
    values: np.ndarray,
    protids: list,
    *,
    resolution: float = DEFAULT_RESOLUTION,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    n_pcs: int = DEFAULT_N_PCS,
    seed: int = DEFAULT_SEED,
) -> Partition:
    """Leiden over a space's features, at one resolution.

    Below three proteins every protein goes into one cluster and scanpy is not
    called, which is what ``leiden_clustering`` does and therefore what the
    agreement test requires.

    Args:
        values: the space's feature matrix, proteins by columns.
        protids: the proteins, in ``values``' row order.
        resolution: Leiden's resolution. Higher gives more clusters.
        n_neighbors: neighbors for the kNN graph, clamped to the cohort.
        n_pcs: principal components the graph is built in, clamped likewise.
        seed: passed to both the neighbor graph and Leiden.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(protids)
    if values.ndim != 2 or values.shape[0] != n:
        raise ClusteringError(
            f"expected a feature matrix with {n} rows, one per protid, got {values.shape}"
        )
    if resolution <= 0:
        raise ClusteringError(f"resolution must be positive, got {resolution}")

    # Both of the above, and the short circuit below, come before the
    # availability check on purpose. A two-protein space has one cluster by
    # definition and needing scanpy installed to say so would make an optional
    # dependency load-bearing for an answer that does not depend on it -- and a
    # malformed call should report *that*, not a missing package.
    if n < 3:
        return Partition(
            protids=list(protids),
            labels=["LC0"] * n,
            resolution=float(resolution),
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            seed=seed,
        )

    available, explanation = is_available()
    if not available:
        raise ClusteringError(explanation)
    import anndata
    import scanpy as sc

    adata = anndata.AnnData(values.copy())
    adata.obs_names = [str(p) for p in protids]
    sc.tl.pca(adata, svd_solver="arpack")
    used_neighbors, used_pcs = _clamped(n, values.shape[1], n_neighbors, n_pcs)
    sc.pp.neighbors(adata, n_neighbors=used_neighbors, n_pcs=used_pcs, random_state=seed)
    sc.tl.leiden(adata, resolution=resolution, random_state=seed)

    codes = adata.obs["leiden"].astype(int)
    width = len(str(int(codes.max())))
    return Partition(
        protids=list(protids),
        labels=["LC" + str(code).zfill(width) for code in codes],
        resolution=float(resolution),
        n_neighbors=used_neighbors,
        n_pcs=used_pcs,
        seed=seed,
    )


def sweep_resolutions(values: np.ndarray, protids: list, resolutions, **kwargs) -> dict:
    """``{resolution: labels}`` for :func:`diagnostics.partition.resolution_sweep`.

    Ordered by resolution rather than by the order they were requested in, so
    that "adjacent" in the sweep means adjacent in the parameter.
    """
    return {
        float(resolution): leiden_partition(values, protids, resolution=resolution, **kwargs).labels
        for resolution in sorted(float(r) for r in resolutions)
    }
