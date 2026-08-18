#!/usr/bin/env python
"""The numerical core of dimensionality reduction, shared by both paths.

``dim_reduction.py`` and the multi-space machinery both call these functions, so
the legacy map and a new space's map are produced by the same code and cannot
drift apart. That sharing is the entire point of the port: two implementations
of "the pipeline's PCA" would eventually disagree, and the disagreement would
show up as a scientific difference nobody could explain.

Behavior here reproduces ``dim_reduction.py`` exactly, including the small-N
clamping added in PR #103 and the column naming, which downstream code depends
on positionally (``plot_interactive.py`` reads its axes from ``df.columns[1]``
and ``[2]``, and derives an output filename from the first of those with digits
stripped). The parity test is what holds that claim up.

Two details that look like mistakes and are not, both preserved deliberately:

* PCA columns are named ``PC0, PC1, ...`` -- zero-based -- while UMAP and t-SNE
  columns are one-based. Changing either would rename a column in
  ``aggregated_features.tsv``.
* ``svd_solver`` is pinned to ``"full"``. The default ``"auto"`` switches to a
  randomized solver above 500 rows or columns and draws from the global numpy
  random state, which is what made every map above ~500 proteins irreproducible
  before PR #106. It is O(N^3); see ADR 0004.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ReducerResult",
    "reduce_pca",
    "reduce_tsne",
    "reduce_umap",
    "small_n_layout",
]


@dataclass(frozen=True)
class ReducerResult:
    """Coordinates, the protids they belong to, and what was actually run.

    ``params_used`` records parameters *after* clamping. A run at N=4 that asked
    for 80 neighbours and got 3 should say so in its manifest, because otherwise
    two runs with identical configs and different N look identical in
    provenance and are not.
    """

    coordinates: np.ndarray
    protids: list
    column_names: list
    params_used: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.coordinates.shape[0] != len(self.protids):
            raise ValueError(
                f"{len(self.protids)} protids but coordinates has "
                f"{self.coordinates.shape[0]} rows"
            )
        if self.coordinates.shape[1] != len(self.column_names):
            raise ValueError(
                f"{len(self.column_names)} column names but coordinates has "
                f"{self.coordinates.shape[1]} columns"
            )

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(
            self.coordinates,
            index=pd.Index(self.protids, name="protid"),
            columns=self.column_names,
        )


def reduce_pca(
    values: np.ndarray,
    protids: list,
    *,
    n_components: int = 2,
    random_state: int = 123456,
    **kwargs,
) -> ReducerResult:
    """Exact, deterministic PCA.

    ``n_components`` is clamped to ``min(values.shape)`` with a warning, matching
    the legacy behavior for a matrix narrower or shorter than the requested
    dimensionality.
    """
    from sklearn.decomposition import PCA

    max_n_components = min(values.shape)
    if max_n_components < 1:
        raise ValueError("empty similarity matrix")
    requested = n_components
    if n_components > max_n_components:
        print(
            f"Warning: the specified value of `n_components` ({n_components})"
            f"cannot be greater than the number of samples ({max_n_components}),"
            f"so `n_components` will be set to {max_n_components}."
        )
        n_components = max_n_components

    pca = PCA(n_components=n_components, svd_solver="full", random_state=random_state, **kwargs)
    coordinates = pca.fit_transform(values)
    return ReducerResult(
        coordinates=coordinates,
        protids=list(protids),
        # Zero-based, matching the legacy naming. See the module docstring.
        column_names=[f"PC{i}" for i in range(coordinates.shape[1])],
        params_used={
            "reducer": "pca",
            "n_components_requested": requested,
            "n_components_used": n_components,
            "svd_solver": "full",
            "random_state": random_state,
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        },
    )


def reduce_tsne(
    values: np.ndarray,
    protids: list,
    *,
    n_components: int = 2,
    perplexity: float = 50,
    n_iter: int = 2000,
    random_state: int = 123456,
    **kwargs,
) -> ReducerResult:
    """t-SNE, with sklearn's perplexity < n_samples constraint respected.

    Clamping is only for that constraint. It deliberately does *not* scale
    perplexity for every small N -- doing so would shift ordinary maps, taking
    N=100 from 50 to 21.
    """
    from sklearn.manifold import TSNE

    n = values.shape[0]
    column_names = [f"tSNE{i + 1}" for i in range(n_components)]

    if n < 2:
        return ReducerResult(
            coordinates=np.zeros((n, n_components), dtype=float),
            protids=list(protids),
            column_names=column_names,
            params_used={"reducer": "tsne", "degenerate": True, "n": n},
        )

    perplexity_check = min(perplexity, max(1, n - 1))
    perplexity_check = min(perplexity_check, n - 1)
    components_used = min(n_components, max(1, n - 1))

    tsne = TSNE(
        n_components=components_used,
        perplexity=float(perplexity_check),
        n_iter=n_iter,
        random_state=random_state,
        **kwargs,
    )
    coordinates = tsne.fit_transform(values)
    if coordinates.shape[1] < n_components:
        pad = np.zeros((n, n_components - coordinates.shape[1]))
        coordinates = np.hstack([coordinates, pad])

    return ReducerResult(
        coordinates=coordinates[:, :n_components],
        protids=list(protids),
        column_names=column_names,
        params_used={
            "reducer": "tsne",
            "n_components_requested": n_components,
            "n_components_used": components_used,
            "perplexity_requested": perplexity,
            "perplexity_used": float(perplexity_check),
            "n_iter": n_iter,
            "random_state": random_state,
        },
    )


def small_n_layout(
    values: np.ndarray,
    protids: list,
    n_components: int,
    prefix: str,
    input_column_names: list | None = None,
) -> ReducerResult:
    """A PCA or identity layout for N too small for UMAP or t-SNE.

    ``input_column_names`` is not decoration. When the input already *is* a PCA
    matrix -- which it is on the ``pca_umap`` and ``pca_tsne`` paths, where PCA
    runs first -- the legacy code reuses the existing ``PC`` columns rather than
    running PCA a second time on PCA output. Dropping that would change the
    coordinates for N < 3, silently, in the one regime nobody looks at.
    """
    n = values.shape[0]
    if n == 0:
        raise ValueError("empty matrix")
    if n == 1:
        coordinates = np.zeros((1, n_components), dtype=float)
    else:
        pc_columns = [
            i for i, name in enumerate(input_column_names or []) if str(name).startswith("PC")
        ]
        if len(pc_columns) >= n_components:
            coordinates = values[:, pc_columns[:n_components]].astype(float)
        else:
            from sklearn.decomposition import PCA

            k = min(n_components, n, values.shape[1])
            reduced = PCA(n_components=k).fit_transform(values)
            if reduced.shape[1] < n_components:
                pad = np.zeros((n, n_components - reduced.shape[1]))
                coordinates = np.hstack([reduced, pad])
            else:
                coordinates = reduced[:, :n_components]
    return ReducerResult(
        coordinates=coordinates,
        protids=list(protids),
        column_names=[f"{prefix}{i + 1}" for i in range(n_components)],
        params_used={"reducer": "small_n_layout", "n": n},
    )


def reduce_umap(
    values: np.ndarray,
    protids: list,
    *,
    n_components: int = 2,
    n_neighbors: int = 80,
    min_dist: float = 0.5,
    random_state: int = 123456,
    input_column_names: list | None = None,
    **kwargs,
) -> ReducerResult:
    """UMAP, falling back to a PCA/identity layout below N=3.

    umap-learn requires ``2 <= n_neighbors < n``, so below three points there is
    no graph to embed and the fallback is the only thing that produces
    coordinates at all.
    """
    n = values.shape[0]

    if n < 3:
        print(
            f"[dim_reduction] N={n} too small for UMAP; writing PCA/identity layout "
            f"as umap coordinates",
            flush=True,
        )
        return small_n_layout(values, protids, n_components, "UMAP", input_column_names)

    from umap import UMAP

    neighbors_check = max(2, min(int(n_neighbors), n - 1))
    umap_fxn = UMAP(
        n_components=n_components,
        random_state=random_state,
        n_neighbors=neighbors_check,
        min_dist=min_dist,
        **kwargs,
    )
    coordinates = umap_fxn.fit_transform(values)
    return ReducerResult(
        coordinates=coordinates,
        protids=list(protids),
        column_names=[f"UMAP{i + 1}" for i in range(coordinates.shape[1])],
        params_used={
            "reducer": "umap",
            "n_components": n_components,
            "n_neighbors_requested": n_neighbors,
            "n_neighbors_used": neighbors_check,
            "min_dist": min_dist,
            "random_state": random_state,
        },
    )
