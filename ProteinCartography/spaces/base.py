#!/usr/bin/env python
"""The Block / Space / View core types.

A **Block** is one representation of the proteins: either a feature matrix
``X`` of shape ``(N, D)`` or a precomputed pairwise distance. A **Space** is a
geometry built from one or more blocks. A **View** renders a space and may never
change it.

Nothing in this module computes anything. It defines the contract that providers
implement and that the registry discovers, so that a new representation can be
added without editing any existing file.

See ``docs/adr/0001-block-space-view.md``, ``docs/adr/0002-fusion-taxonomy-and-
normalization.md`` and ``docs/adr/0003-the-fusable-flag.md``.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "BLOCK_KINDS",
    "CHANNEL_SEMANTICS",
    "METRICS",
    "NORMALIZATIONS",
    "STRATEGIES",
    "SYMMETRIZATIONS",
    "BlockProvider",
    "BlockResult",
    "BlockSpec",
    "BlockSpecError",
    "NotFusableError",
    "SpaceSpec",
    "block_spec_error",
]

BLOCK_KINDS = ("features", "pairwise", "pairwise_directed")
METRICS = ("euclidean", "cosine", "precomputed", "jaccard")
NORMALIZATIONS = ("none", "zscore_within", "unit_mean_distance")
STRATEGIES = ("none", "early", "late", "graph", "coregistered")

#: How a symmetric pairwise block was derived from an asymmetric measurement.
#: TM-score is length-normalized per query, so TM(a->b) != TM(b->a) whenever the
#: two proteins differ in length -- measured on production data, only 40% of
#: both-measured pairs are exactly equal. Collapsing that to one number is a
#: modelling choice and has to be recorded, not assumed.
SYMMETRIZATIONS = ("min", "max", "mean", "query_normalized", "not_applicable")

#: Per-cell annotation channels, with fixed meaning and polarity.
#:
#: These arrays are written to disk. If their interpretation is not fixed here,
#: it cannot be recovered from the file later -- which is precisely the failure
#: ADR 0009 documents for the similarity matrix's `0.0` fill.
CHANNEL_SEMANTICS = {
    "censored": {
        "dtype": "bool",
        "polarity": "True means NOT measured.",
        "meaning": (
            "the value was not reported by the upstream tool -- for the TM matrix, "
            "the pair lost the per-query top-N cut. Recoverable in principle by "
            "rerunning with a larger cap. See ADR 0009."
        ),
    },
    "absent": {
        "dtype": "bool",
        "polarity": "True means NO VALUE EXISTS.",
        "meaning": (
            "the provider genuinely has no value for this protein -- a predictor "
            "that failed or does not apply. Distinct from 'censored': rerunning "
            "will not produce one."
        ),
    },
    "confidence": {
        "dtype": "float",
        "polarity": "1.0 means fully confident, 0.0 means no confidence.",
        "meaning": (
            "the provider's own confidence in this value, in [0, 1]. Carried "
            "separately so that a low-confidence prediction does not silently "
            "move a point, which is what encoding confidence into the value "
            "itself would do."
        ),
    },
}


class BlockSpecError(ValueError):
    """Raised when a block or space specification is internally inconsistent."""


class NotFusableError(ValueError):
    """Raised when a block that must stay overlay-only is used in a geometry.

    The message always states *why*, because the reason is the whole point --
    see ADR 0003. A future maintainer who hits this should find an argument, not
    an obstacle.
    """


def block_spec_error(field_path: str, value, allowed) -> BlockSpecError:
    return BlockSpecError(
        f"{field_path}: {value!r} is not valid. Allowed: {', '.join(sorted(allowed))}."
    )


@dataclass(frozen=True)
class BlockSpec:
    """What a block *is*, independent of any particular computation of it.

    Attributes:
        id: unique within a config. Used as the on-disk directory name.
        kind: ``"features"`` for an ``(N, D)`` matrix, ``"pairwise"`` for a
            precomputed distance.
        fusable: whether this block may enter a geometry at all. False means
            overlay-only; see ADR 0003.
        metric: how distances are computed from the features. ``"precomputed"``
            for a pairwise block.
        normalization: applied before any weighting. ADR 0002 requires that
            block scale be normalized before weights are meaningful.
        provider: the registry name that produces this block.
        params: provider-specific parameters, validated by the provider itself
            at the top of ``compute``. Nothing calls ``spec_schema`` for it --
            see the note on that attribute.
        not_fusable_reason: required when ``fusable`` is False. This string is
            shown to the user verbatim when they try to fuse the block.
        version: bumped by a provider when its output changes meaning. Recorded
            on the block and, today, read by nothing: ``Manifest.cache_key``
            excludes ``derived``, which is where this lands, so a bump does not
            invalidate a cached block (FOLLOWUPS #46).
    """

    id: str
    kind: str
    fusable: bool
    metric: str
    normalization: str
    provider: str
    params: dict = field(default_factory=dict)
    not_fusable_reason: str | None = None
    version: str = "1"
    symmetrization: str | None = None
    distance_metric: str | None = None

    def __post_init__(self):
        if not self.id or not isinstance(self.id, str):
            raise BlockSpecError(f"block id must be a non-empty string, got {self.id!r}")
        if self.kind not in BLOCK_KINDS:
            raise block_spec_error(f"blocks.{self.id}.kind", self.kind, BLOCK_KINDS)
        if self.metric not in METRICS:
            raise block_spec_error(f"blocks.{self.id}.metric", self.metric, METRICS)
        if self.normalization not in NORMALIZATIONS:
            raise block_spec_error(
                f"blocks.{self.id}.normalization", self.normalization, NORMALIZATIONS
            )
        if self.kind in ("pairwise", "pairwise_directed") and self.metric != "precomputed":
            raise BlockSpecError(
                f"blocks.{self.id}: a pairwise block carries its own distances, so its "
                f"metric must be 'precomputed', not {self.metric!r}. What produced "
                "those distances belongs in `distance_metric`."
            )
        if self.symmetrization is not None and self.symmetrization not in SYMMETRIZATIONS:
            raise block_spec_error(
                f"blocks.{self.id}.symmetrization", self.symmetrization, SYMMETRIZATIONS
            )
        if self.kind == "pairwise" and self.symmetrization is None:
            raise BlockSpecError(
                f"blocks.{self.id}: a symmetric pairwise block must declare how it was "
                f"symmetrized. Allowed: {', '.join(sorted(SYMMETRIZATIONS))}. "
                "TM-score is length-normalized per query, so TM(a->b) != TM(b->a) for "
                "any pair of different lengths -- on production data only 40% of "
                "both-measured pairs are exactly equal. Collapsing that to one number "
                "is a modelling choice, and an unrecorded modelling choice is not "
                "reproducible. Use 'not_applicable' if the source really is symmetric, "
                "or kind='pairwise_directed' to keep both directions."
            )
        if self.kind == "pairwise_directed" and self.symmetrization not in (
            None,
            "not_applicable",
        ):
            raise BlockSpecError(
                f"blocks.{self.id}: a directed pairwise block keeps both directions, so "
                f"it has not been symmetrized; symmetrization={self.symmetrization!r} "
                "is contradictory."
            )
        if self.kind == "features" and self.metric == "precomputed":
            raise BlockSpecError(
                f"blocks.{self.id}: metric 'precomputed' means the block supplies "
                "distances directly, so kind must be 'pairwise'."
            )
        if not self.fusable and not self.not_fusable_reason:
            raise BlockSpecError(
                f"blocks.{self.id}: a block marked fusable=False must carry a "
                "not_fusable_reason. The reason is shown to users who try to fuse "
                "it, and is the only thing that stops the flag being removed later "
                "as an obstacle. See ADR 0003."
            )

    def require_fusable(self, space_id: str) -> None:
        """Raise :class:`NotFusableError` if this block may not enter a geometry."""
        if self.fusable:
            return
        raise NotFusableError(
            "\n".join(
                [
                    f"Block {self.id!r} cannot be fused into space {space_id!r}.",
                    f"Reason: {self.not_fusable_reason}",
                    "",
                    f"{self.id!r} is available as an overlay on any space, which "
                    "shows the same information without letting it move the points.",
                ]
            )
        )


@dataclass(frozen=True)
class BlockResult:
    """A computed block: values, the protids they belong to, and provenance.

    ``protids`` is the canonical order and the only thing that gives the rows
    meaning. Nothing here hands back an unlabeled array.

    **Per-cell annotations go in named channels, never in one anonymous mask.**
    An earlier revision had a single ``mask`` field. It had no declared polarity
    -- ``numpy.ma`` reads True as invalid, pandas and scikit-learn read True as
    valid -- and no declared meaning, so three different providers would have
    filled the same slot with three different ideas: "Foldseek did not report
    this pair", "this predictor has no value for this protein", and "this
    prediction has low confidence". Those are different facts with different
    consequences, and the array is written to disk, where the ambiguity becomes
    permanent.

    So each channel is named, its polarity is fixed by :data:`CHANNEL_SEMANTICS`,
    and unknown names are rejected. ``mask=`` is still accepted and is recorded
    as ``censored``, which is what it always meant here.
    """

    spec: BlockSpec
    protids: list
    features: np.ndarray | None = None
    distances: np.ndarray | None = None
    channels: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    #: The censoring channel, under its historical name. Kept in sync with
    #: ``channels["censored"]``; ``channels`` is the authoritative form.
    mask: np.ndarray | None = None

    def __post_init__(self):
        if self.mask is not None:
            if "censored" in self.channels:
                raise BlockSpecError(
                    f"blocks.{self.spec.id}: both mask= and channels['censored'] were "
                    "given. mask= is the old spelling of the censored channel; pass "
                    "one or the other."
                )
            object.__setattr__(self, "channels", {**self.channels, "censored": self.mask})
        elif "censored" in self.channels:
            object.__setattr__(self, "mask", self.channels["censored"])

        if self.features is None and self.distances is None:
            raise BlockSpecError(
                f"blocks.{self.spec.id}: a BlockResult must carry either features " "or distances."
            )
        if self.features is not None and self.distances is not None:
            raise BlockSpecError(
                f"blocks.{self.spec.id}: a BlockResult carries features or "
                "distances, not both. The distances are derived from the features "
                "by the space's metric."
            )
        n = len(self.protids)
        if self.spec.kind == "features":
            if self.features is None:
                raise BlockSpecError(
                    f"blocks.{self.spec.id}: kind is 'features' but no features " "were supplied."
                )
            if not np.issubdtype(self.features.dtype, np.floating):
                raise BlockSpecError(
                    f"blocks.{self.spec.id}: features must be a floating-point array, "
                    f"got dtype {self.features.dtype}. A ragged or object array is "
                    "accepted by numpy here but fails later when the block is stored, "
                    "which is a long way from the mistake. Encode variable-length data "
                    "as a fixed-width numeric array, or supply precomputed distances "
                    "with kind='pairwise'."
                )
            if self.features.shape[0] != n:
                raise BlockSpecError(
                    f"blocks.{self.spec.id}: {n} protids but features has "
                    f"{self.features.shape[0]} rows."
                )
        elif self.spec.kind == "pairwise":
            if self.distances is None:
                raise BlockSpecError(
                    f"blocks.{self.spec.id}: kind is 'pairwise' but no distances " "were supplied."
                )
            expected = n * (n - 1) // 2
            if self.distances.ndim != 1 or self.distances.shape[0] != expected:
                raise BlockSpecError(
                    f"blocks.{self.spec.id}: pairwise distances must be the "
                    f"condensed upper triangle, so shape ({expected},) for {n} "
                    f"proteins, but got {self.distances.shape}. Use "
                    "scipy.spatial.distance.squareform to condense a square matrix, "
                    "or kind='pairwise_directed' to keep both directions."
                )
        else:  # pairwise_directed
            if self.distances is None:
                raise BlockSpecError(
                    f"blocks.{self.spec.id}: kind is 'pairwise_directed' but no "
                    "distances were supplied."
                )
            if self.distances.shape != (n, n):
                raise BlockSpecError(
                    f"blocks.{self.spec.id}: a directed pairwise block keeps both "
                    f"directions, so distances must be square ({n}, {n}), but got "
                    f"{self.distances.shape}."
                )

        target = self.features if self.features is not None else self.distances
        for name, array in self.channels.items():
            if name not in CHANNEL_SEMANTICS:
                raise BlockSpecError(
                    f"blocks.{self.spec.id}: unknown channel {name!r}. Known channels: "
                    f"{', '.join(sorted(CHANNEL_SEMANTICS))}. A channel's meaning and "
                    "polarity have to be declared, because the array is written to "
                    "disk and its interpretation cannot be recovered later."
                )
            semantics = CHANNEL_SEMANTICS[name]
            if array.shape != target.shape:
                raise BlockSpecError(
                    f"blocks.{self.spec.id}: channel {name!r} has shape {array.shape}, "
                    f"which does not match the data shape {target.shape}."
                )
            if semantics["dtype"] == "bool" and array.dtype != np.bool_:
                raise BlockSpecError(
                    f"blocks.{self.spec.id}: channel {name!r} must be boolean, got "
                    f"{array.dtype}. {semantics['polarity']} A float here invites "
                    "arithmetic that silently treats a flag as a number."
                )
            if semantics["dtype"] == "float":
                if not np.issubdtype(array.dtype, np.floating):
                    raise BlockSpecError(
                        f"blocks.{self.spec.id}: channel {name!r} must be floating "
                        f"point, got {array.dtype}."
                    )
                if array.size and (np.nanmin(array) < 0 or np.nanmax(array) > 1):
                    raise BlockSpecError(
                        f"blocks.{self.spec.id}: channel {name!r} must lie in [0, 1], "
                        f"but ranges [{np.nanmin(array)}, {np.nanmax(array)}]."
                    )

        # A NaN with no channel explaining it is a silent hole. Refuse it: the
        # alternative is a coordinate computed from a value nobody supplied.
        if target is not None and np.issubdtype(target.dtype, np.floating):
            nans = np.isnan(target)
            if nans.any():
                explained = np.zeros(target.shape, dtype=bool)
                for name in ("censored", "absent"):
                    if name in self.channels:
                        explained |= self.channels[name]
                unexplained = int((nans & ~explained).sum())
                if unexplained:
                    raise BlockSpecError(
                        f"blocks.{self.spec.id}: {unexplained} value(s) are NaN but are "
                        "not covered by a 'censored' or 'absent' channel. An "
                        "unexplained NaN becomes a coordinate for a protein that was "
                        "never measured. Declare which it is."
                    )

    @property
    def n_proteins(self) -> int:
        return len(self.protids)

    @property
    def n_features(self) -> int | None:
        return None if self.features is None else int(self.features.shape[1])

    @property
    def censoring_rate(self) -> float | None:
        """Fraction of cells not measured, or None if the block never said.

        None rather than 0.0 on purpose. A block that declared no censoring
        channel has made no claim about censoring, and reporting 0.0 would put
        that non-claim into the manifest as a measurement.
        """
        censored = self.channels.get("censored")
        return None if censored is None else float(censored.mean())

    @property
    def absent_rate(self) -> float | None:
        """Fraction of cells for which no value exists, or None if unstated."""
        absent = self.channels.get("absent")
        return None if absent is None else float(absent.mean())

    def channel(self, name: str) -> np.ndarray | None:
        return self.channels.get(name)

    def with_protids(self, protids: list) -> BlockResult:
        """Relabel the rows.

        This only renames; it does not reorder the data. Handing it a permuted
        list would silently attach every protein's values to a different
        protein, which is the exact failure this design exists to prevent, so it
        refuses anything but a same-length, duplicate-free list. To *reorder*,
        go through :meth:`index.ProteinIndex.align`, which moves the values too.
        """
        protids = list(protids)
        if len(protids) != len(self.protids):
            raise BlockSpecError(
                f"blocks.{self.spec.id}: with_protids got {len(protids)} labels for "
                f"{len(self.protids)} rows. It renames rows in place and never "
                "reorders them; use ProteinIndex.align to reorder."
            )
        if len(set(protids)) != len(protids):
            duplicates = sorted({p for p in protids if protids.count(p) > 1})
            raise BlockSpecError(
                f"blocks.{self.spec.id}: with_protids got duplicate labels "
                f"{duplicates[:5]}. A duplicate protid doubles that protein's weight "
                "in every geometry built from this block."
            )
        return replace(self, protids=protids)


@runtime_checkable
class BlockProvider(Protocol):
    """What a representation must implement to be usable as a block.

    Deliberately a Protocol rather than a base class: providers live in
    separate, optionally-installed packages (ADR 0006), and requiring them to
    import this module's class hierarchy would mean importing this package --
    and its dependencies -- before ``is_available`` could even be consulted.
    """

    #: Callable validating and normalizing this provider's params dict.
    #: Raises on bad input. A provider that already depends on pydantic may
    #: implement this with a pydantic model; the contract is the callable.
    #:
    #: **Declarative only -- the framework never calls it.** `config_schema`
    #: validates a config without importing any provider, so it cannot reach
    #: this at parse time. Every built-in provider calls its own
    #: `validate_params` first thing in `compute`, and a provider that does not
    #: gets no validation. See docs/EXTENDING.md §2.
    spec_schema: object

    def is_available(self) -> tuple:
        """Return ``(available, reason)``.

        ``reason`` explains what is missing and how to get it when
        ``available`` is False. A missing optional dependency must degrade the
        run, never break it -- the Snakefile skips unavailable spaces with a log
        line. See ADR 0006.
        """
        ...

    def compute(self, ctx, params: dict) -> BlockResult:
        ...


@dataclass(frozen=True)
class SpaceSpec:
    """A geometry: which blocks, combined how, reduced by what."""

    id: str
    blocks: tuple
    strategy: str = "none"
    weights: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    reducers: tuple = ("pca_umap",)

    def __post_init__(self):
        if not self.id:
            raise BlockSpecError("a space must have an id")
        if not self.blocks:
            raise BlockSpecError(f"spaces.{self.id}: at least one block is required")
        if self.strategy not in STRATEGIES:
            raise block_spec_error(f"spaces.{self.id}.strategy", self.strategy, STRATEGIES)
        if self.strategy == "none" and len(self.blocks) != 1:
            raise BlockSpecError(
                f"spaces.{self.id}: strategy 'none' means a single block, but "
                f"{len(self.blocks)} were given. Choose a fusion strategy, or split "
                "these into separate co-registered spaces."
            )
        if len(set(self.blocks)) != len(self.blocks):
            raise BlockSpecError(f"spaces.{self.id}: a block is listed more than once")
        unknown = set(self.weights) - set(self.blocks)
        if unknown:
            raise BlockSpecError(
                f"spaces.{self.id}: weights given for block(s) not in this space: "
                f"{sorted(unknown)}"
            )
        for name, w in self.weights.items():
            if not isinstance(w, (int, float)) or isinstance(w, bool):
                raise BlockSpecError(
                    f"spaces.{self.id}.weights.{name}: expected a number, got {w!r}"
                )
            if w < 0:
                raise BlockSpecError(
                    f"spaces.{self.id}.weights.{name}: a negative weight ({w}) has no "
                    "meaning -- distances cannot be subtracted from one another."
                )

    @property
    def is_multiblock(self) -> bool:
        return len(self.blocks) > 1

    def weight_for(self, block_id: str) -> float:
        return float(self.weights.get(block_id, 1.0))
