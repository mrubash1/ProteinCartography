#!/usr/bin/env python
"""The TM-score block: the pipeline's existing representation, as a block.

This wraps `all_by_all_tmscore_pivoted.tsv` -- the artifact the pipeline has
always produced -- so that the existing map and any new one are built from the
same object. It computes nothing new; `foldseek_clustering.py` still produces
the matrix exactly as before.

Two representations, and the difference between them matters more than it looks.

``profile`` (the default, and what the pipeline has always done) treats each
protein's *row* of the similarity matrix as its feature vector, and distance
between proteins is Euclidean distance between rows. This is safe on a matrix
whose columns are permuted, because every row shares the same permutation and
Euclidean distance is invariant to a consistent reordering of coordinates. That
invariance is the only reason the shipped UMAP is not garbage on pre-#106
output.

``direct`` treats the matrix as a matrix: distance is ``1 - TM`` for the pair.
It is the more natural reading and it is **silently catastrophic** on a permuted
matrix, because it reads cell ``(i, j)`` expecting protein *i* against protein
*j*. Measured on a production run, that is the wrong cell 99.92% of the time. So
``direct`` is gated behind the alignment assertion and an explicit config flag,
and it must declare how it symmetrized -- TM-score is length-normalized per
query, so ``TM(a->b) != TM(b->a)`` and there is no symmetric matrix to read
without making a choice.

See ``docs/adr/0007-matrix-index-alignment.md`` and
``docs/adr/0009-censoring-semantics.md``.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field, replace

import numpy as np
from diagnostics.metricity import metricity_report
from matrix_io import load_labeled_matrix, summarize_censoring
from spaces.base import BlockResult, BlockSpec
from spaces.manifest import Manifest, file_digest

__all__ = ["MATRIX_FILENAME", "PipelineContext", "TMScoreProvider"]

MATRIX_FILENAME = "all_by_all_tmscore_pivoted.tsv"
CLUSTERING_SUBDIR = "foldseek_clustering_results"

VALID_REPRESENTATIONS = ("profile", "direct")
VALID_SYMMETRIZATIONS = ("min", "max", "mean")


@dataclass(frozen=True)
class PipelineContext:
    """What a provider is given about the run it is part of.

    Deliberately thin. A provider that needs more should take it as a parameter,
    so that what it depends on is visible in the config rather than reached for
    through a context object.
    """

    output_dir: str
    seed: int = 123456
    extras: dict = field(default_factory=dict)

    def path(self, *parts) -> str:
        return os.path.join(self.output_dir, *parts)


def validate_params(params: dict) -> dict:
    """Validate and normalize this provider's parameters."""
    params = dict(params or {})
    representation = params.get("representation", "profile")
    if representation not in VALID_REPRESENTATIONS:
        raise ValueError(
            f"tmscore.representation: {representation!r} is not valid. "
            f"Allowed: {', '.join(VALID_REPRESENTATIONS)}."
        )
    if representation == "direct":
        if params.get("alignment_verified", False) is not True:
            raise ValueError(
                "tmscore.representation 'direct' reads the similarity matrix as a "
                "matrix, so it requires verified row/column alignment. Set "
                "alignment_verified: true only once matrix_io accepts the matrix "
                "without repair. 'profile' is safe either way."
            )
        symmetrization = params.get("symmetrization", "mean")
        if symmetrization not in VALID_SYMMETRIZATIONS:
            raise ValueError(
                f"tmscore.symmetrization: {symmetrization!r} is not valid. "
                f"Allowed: {', '.join(VALID_SYMMETRIZATIONS)}. TM-score is "
                "length-normalized per query, so the two directions genuinely "
                "differ and one of them has to be chosen."
            )
        params["symmetrization"] = symmetrization
    params["representation"] = representation
    return params


def _symmetrize(values: np.ndarray, rule: str) -> np.ndarray:
    transposed = values.T
    if rule == "min":
        return np.minimum(values, transposed)
    if rule == "max":
        return np.maximum(values, transposed)
    return (values + transposed) / 2.0


class TMScoreProvider:
    """Produces the `tmscore` block from the pivoted similarity matrix."""

    spec_schema = staticmethod(validate_params)

    #: Bumped when the block's meaning changes, so cached blocks invalidate.
    version = "1"

    def __init__(self, matrix_path: str | None = None):
        self._matrix_path = matrix_path

    # -- discovery ---------------------------------------------------------

    def matrix_path_for(self, ctx: PipelineContext, params: dict) -> str:
        if params.get("matrix_path"):
            return params["matrix_path"]
        if self._matrix_path:
            return self._matrix_path
        return ctx.path(CLUSTERING_SUBDIR, MATRIX_FILENAME)

    def is_available(self) -> tuple:
        """Always available: it depends only on foldseek, already a dependency.

        The matrix file's absence is a missing *input*, not a missing
        dependency, and is reported when the block is computed rather than here
        -- otherwise a survey run before the pipeline has produced the matrix
        would report the provider as broken.
        """
        return True, ""

    # -- compute -----------------------------------------------------------

    def compute(self, ctx: PipelineContext, params: dict) -> BlockResult:
        params = validate_params(params)
        path = self.matrix_path_for(ctx, params)
        representation = params["representation"]

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"the tmscore block needs {path}, which the `foldseek_clustering` "
                "rule produces. Run the pipeline up to that rule first."
            )

        # `profile` tolerates a permuted matrix, but there is no reason to accept
        # one silently: repair it and say so. `direct` cannot tolerate it at all,
        # and its config gate has already required the alignment be verified, so
        # here it is asserted rather than repaired.
        matrix = load_labeled_matrix(
            path,
            require_alignment=True,
            repair=(representation == "profile"),
        )

        censoring = summarize_censoring(matrix)
        manifest = Manifest.build(
            "block",
            params.get("block_id", "tmscore"),
            provider="tmscore",
            params=params,
            inputs={"similarity_matrix": file_digest(path)},
            protids=matrix.protids,
            seed=ctx.seed,
            extra={"censoring": censoring},
        )

        if representation == "profile":
            return self._profile_block(matrix, params, manifest)
        return self._direct_block(matrix, params, manifest, censoring)

    def _profile_block(self, matrix, params, manifest) -> BlockResult:
        spec = BlockSpec(
            id=params.get("block_id", "tmscore"),
            kind="features",
            fusable=True,
            metric="euclidean",
            normalization=params.get("normalization", "unit_mean_distance"),
            provider="tmscore",
            params=params,
            version=self.version,
        )
        return BlockResult(
            spec=spec,
            protids=list(matrix.protids),
            features=np.ascontiguousarray(matrix.values, dtype=np.float32),
            channels={"censored": np.ascontiguousarray(matrix.censored)},
            manifest=manifest.to_dict(),
        )

    def _direct_block(self, matrix, params, manifest, censoring) -> BlockResult:
        from scipy.spatial.distance import squareform

        rule = params["symmetrization"]
        similarity = _symmetrize(matrix.values.astype(np.float64), rule)
        # A pair is censored if either direction was missing: the symmetrized
        # value is only as trustworthy as its weaker input.
        censored_square = matrix.censored | matrix.censored.T
        np.fill_diagonal(censored_square, False)

        distances = 1.0 - similarity
        np.fill_diagonal(distances, 0.0)
        # squareform needs exact symmetry; floating-point averaging can leave a
        # last-bit asymmetry that trips its check.
        distances = (distances + distances.T) / 2.0

        # This is the only point in the pipeline where a full square distance
        # matrix exists, so it is the only point where its metricity can be
        # measured. The config gate for `direct` checks that the rows and
        # columns line up; that says nothing about whether `1 - TM` can be
        # embedded in a Euclidean space, and largely it cannot. Recorded on the
        # block rather than on the space because it is a property of these
        # distances, and because no space consumes a pairwise block yet --
        # `reduce_space.read_blocks` refuses one for want of a metric-aware
        # reducer. When one lands, the number is already attached to the input.
        #
        # It goes in `derived` and not in `extra`, because `extra` is folded
        # into `cache_key` and `derived` is not. A cache key is an identity of
        # the *inputs*; a figure computed from the output could only be supplied
        # by a caller who had already built the block, which is exactly the
        # caller the cache is not for.
        manifest = replace(
            manifest,
            derived={
                **manifest.derived,
                "metricity": metricity_report(
                    distances,
                    censoring_rate=censoring.get("censoring_rate"),
                    n_proteins=len(matrix.protids),
                ),
            },
        )

        spec = BlockSpec(
            id=params.get("block_id", "tmscore"),
            kind="pairwise",
            fusable=True,
            metric="precomputed",
            normalization=params.get("normalization", "unit_mean_distance"),
            provider="tmscore",
            params=params,
            version=self.version,
            symmetrization=rule,
            distance_metric="1 - TM-score",
        )
        return BlockResult(
            spec=spec,
            protids=list(matrix.protids),
            distances=squareform(distances, checks=False).astype(np.float32),
            channels={"censored": squareform(censored_square, checks=False).astype(bool)},
            manifest=manifest.to_dict(),
        )


def register() -> None:
    """Register this provider as a built-in.

    Built-ins exist so the pipeline works from a source checkout with nothing
    installed, which is how snakemake invokes it. Entry points still take
    precedence, so a third party can replace this implementation.
    """
    from spaces.registry import BLOCK_GROUP, register_builtin

    register_builtin(BLOCK_GROUP, "tmscore", TMScoreProvider)
