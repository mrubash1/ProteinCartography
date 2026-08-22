#!/usr/bin/env python
"""Validated configuration for blocks, spaces, and fusion.

Frozen dataclasses with explicit validation, no third-party schema library. The
reasoning is in ``docs/adr/0010-config-validation-without-pydantic.md``: the
validator runs inside the snakemake driver environment, that environment is
closed to new dependencies for this work, and pydantic v2 is a compiled
dependency.

Two things here are load-bearing rather than housekeeping:

* :func:`from_legacy` means an existing ``config.yml`` keeps working untouched.
  Every new key has a default that reproduces current behavior.
* The validator rejects a ``fusable: false`` block in a multi-block space, with
  an error that states *why*. That is the entire enforcement mechanism for
  ADR 0003, and the reason text is the part that matters -- a future maintainer
  should meet an argument, not an obstacle.
"""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from diagnostics.partition import CONTROL_DESCRIPTIONS
from enrichment import ENCODINGS
from fusion import STRATEGY_PARAMS
from spaces.base import (
    METRICS,
    NORMALIZATIONS,
    STRATEGIES,
    NotFusableError,
)

__all__ = [
    "NOT_FUSABLE_PROVIDERS",
    "NOT_FUSABLE_REASONS",
    "SIGNIFICANCE_MEASURES",
    "BlockConfig",
    "CohortConfig",
    "ConfigError",
    "CoregistrationConfig",
    "DiagnosticsConfig",
    "EnrichmentConfig",
    "MultispaceConfig",
    "SpaceConfig",
    "from_legacy",
]

REPRESENTATIONS = ("profile", "direct")

#: Used only where a block's normalization has to be resolved without asking a
#: provider. A block config that leaves `normalization` unset is asking its
#: provider to choose, and the provider is the thing that knows.
DEFAULT_NORMALIZATION = "unit_mean_distance"
SELECTION_RULES = ("as_filtered", "accession", "significance")

#: Each significance measure a cohort can be ranked by, and which direction is
#: better. Consumed by ``cohort.select``; declared here because it is
#: configuration vocabulary, and because validating it at config-parse time is
#: what stops a typo from failing four hours into a search run.
#:
#: ``better`` is the point of the table. Ranking by e-value ascending and by
#: TM-score ascending are both one-word changes and only one of them is right;
#: deriving the direction from the measure's name means a caller holding a raw
#: float cannot get it wrong.
#:
#: ``tmscore`` is listed and is *not* obtainable during a search-mode run: the
#: Foldseek web API reports no TM-score, and the pipeline's TM-scores come from
#: the local all-versus-all run, which happens after the structures the cohort
#: rule chooses have already been downloaded. It stays in the table because the
#: measure is well-defined wherever scores do exist. See ADR 0008.
SIGNIFICANCE_MEASURES = {
    "evalue": {
        "better": "lower",
        "description": "BLAST or Foldseek expectation value; lower is a stronger hit",
    },
    "bits": {
        "better": "higher",
        "description": "alignment bit score; higher is a stronger hit",
    },
    "tmscore": {
        "better": "higher",
        "description": "structural alignment TM-score in [0, 1]; higher is more similar",
    },
}

#: The measure used when a config asks for `significance` without naming one.
DEFAULT_SIGNIFICANCE_MEASURE = "evalue"
LEGACY_PLOTTING_MODES = ("pca", "tsne", "umap", "pca_tsne", "pca_umap")

#: The space that legacy `plotting_modes` maps onto. Named so that an existing
#: config produces a space whose outputs land where they always have.
LEGACY_SPACE_ID = "structure"
LEGACY_BLOCK_ID = "tmscore"

#: Providers whose output must stay overlay-only, keyed by ``provider`` name.
#:
#: Keying on the *provider* rather than the block id is load-bearing. The block
#: id is a free-text key the user chooses, so a table keyed on it protects
#: ``taxonomy:`` and misses ``tax:``, ``Taxonomy:``, and ``lineage:`` -- the
#: protection would apply exactly to users who already knew about it. The
#: provider is what actually determines what the numbers are.
NOT_FUSABLE_PROVIDERS = {
    "uniprot_lineage": "taxonomy",
    "taxonomy": "taxonomy",
    "phylogeny": "phylogeny",
    "noveltree": "phylogeny",
    "iqtree": "phylogeny",
    "patristic": "phylogeny",
    "plddt": "plddt",
    "pdb_confidence": "pdb_confidence",
    "censoring": "censoring",
    "disorder": "disorder",
    "iupred3": "disorder",
    "fldpnn": "disorder",
    "stability_qc": "stability_qc",
    "struclusters": "struclusters",
    "strucluster": "struclusters",
}

#: Why each of those signals may never enter a geometry. See ADR 0003. Held here
#: rather than in each provider so that the argument lives in one place and a
#: config can be validated without importing any provider.
NOT_FUSABLE_REASONS = {
    "taxonomy": (
        "fusing taxonomy makes every taxon-specific cluster claim circular -- the "
        "clusters would separate by taxon because taxon was an axis"
    ),
    "phylogeny": (
        "patristic distance is derived from a tree built on these same sequences, so "
        "fusing it makes every evolutionary claim circular"
    ),
    "plddt": (
        "pLDDT correlates strongly with length, so fusing it makes protein length a "
        "principal axis of a biology map"
    ),
    "pdb_confidence": (
        "prediction confidence correlates strongly with length, so fusing it makes "
        "protein length a principal axis of a biology map"
    ),
    "censoring": (
        "censoring rate is a property of how well a protein was measured, not of the "
        "protein, and it correlates with length"
    ),
    "disorder": (
        "disorder fraction explains most TM-score failures and tracks length, so "
        "fusing it makes length an axis"
    ),
    "stability_qc": (
        "neighborhood stability qualifies a measurement; fusing it into the thing it "
        "qualifies is a category error"
    ),
    "struclusters": (
        "struclusters is a structurally-gated graph clustered on amino-acid identity "
        "-- foldseek clust receives the alignment database and never the TM-score "
        "database -- so using it inside a geometry that is then contrasted against "
        "sequence space is partly circular (ADR 0003)"
    ),
}


class ConfigError(ValueError):
    """Raised when a configuration is invalid. The message names the key path."""


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------


def _require(path: str, condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(f"{path}: {message}")


def _require_type(path: str, value, types, type_name: str):
    if isinstance(value, bool) and bool not in (types if isinstance(types, tuple) else (types,)):
        # bool is an int subclass; accepting True where a number belongs is
        # never what the user meant.
        raise ConfigError(f"{path}: expected {type_name}, got a boolean ({value!r})")
    if not isinstance(value, types):
        raise ConfigError(f"{path}: expected {type_name}, got {type(value).__name__} ({value!r})")
    return value


def _require_str(path: str, value) -> str:
    return _require_type(path, value, str, "a string")


def _require_bool(path: str, value) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path}: expected true or false, got {type(value).__name__} ({value!r})")
    return value


def _require_int(path: str, value) -> int:
    return int(_require_type(path, value, int, "an integer"))


def _require_number(path: str, value) -> float:
    return float(_require_type(path, value, (int, float), "a number"))


def _require_choice(path: str, value, allowed) -> str:
    _require_str(path, value)
    if value not in allowed:
        raise ConfigError(f"{path}: {value!r} is not valid. Allowed: {', '.join(sorted(allowed))}.")
    return value


def _require_mapping(path: str, value) -> dict:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__} ({value!r})")
    return dict(value)


def _require_sequence(path: str, value) -> list:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"{path}: expected a list, got {type(value).__name__} ({value!r})")
    return list(value)


def _reject_unknown_keys(path: str, data: Mapping, known) -> None:
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s) {unknown}. Known keys: {sorted(known)}. "
            "A misspelled key is silently ignored by a permissive loader, which is "
            "how a setting you thought you changed turns out not to have applied."
        )


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortConfig:
    """Which proteins reach the map, and how truncation is recorded (ADR 0008).

    ``selection`` defaults to ``as_filtered``, which is what the pipeline does
    today: truncate the filtered hit list in the order it arrives, which is
    UniProt's response order. That order is *not* reproducible, and the name says
    so rather than implying otherwise -- calling it ``accession`` would repeat
    PR #106's mistake of assuming a sort survived the round-trip through UniProt
    when it does not. ``accession`` and ``significance`` are both reproducible
    and both change which proteins reach the map, so both are opt-in.
    """

    max_structures: int = 5000
    selection: str = "as_filtered"
    significance_rule: dict = field(default_factory=dict)
    record_truncation: bool = True

    def __post_init__(self):
        _require_int("cohort.max_structures", self.max_structures)
        _require(
            "cohort.max_structures",
            self.max_structures > 0,
            f"must be positive, got {self.max_structures}",
        )
        _require_choice("cohort.selection", self.selection, SELECTION_RULES)
        _require_mapping("cohort.significance_rule", self.significance_rule)
        # Checked here rather than where it is used, because where it is used is
        # after the searches have run.
        _require_choice(
            "cohort.significance_rule.measure", self.measure, tuple(SIGNIFICANCE_MEASURES)
        )
        _require_bool("cohort.record_truncation", self.record_truncation)

    @property
    def measure(self) -> str:
        """Which significance measure ranks the cohort."""
        return self.significance_rule.get("measure", DEFAULT_SIGNIFICANCE_MEASURE)

    @classmethod
    def from_dict(cls, data: Mapping | None) -> CohortConfig:
        data = _require_mapping("cohort", data or {})
        known = {"max_structures", "selection", "significance_rule", "record_truncation"}
        _reject_unknown_keys("cohort", data, known)
        return cls(
            max_structures=data.get("max_structures", 5000),
            selection=data.get("selection", "as_filtered"),
            significance_rule=dict(data.get("significance_rule", {}) or {}),
            record_truncation=data.get("record_truncation", True),
        )


@dataclass(frozen=True)
class BlockConfig:
    """One representation."""

    id: str
    provider: str
    params: dict = field(default_factory=dict)
    fusable: bool = True
    not_fusable_reason: str | None = None
    #: ``None`` means "whatever this provider says", which is not the same as
    #: any particular value. A block's features decide what normalizing them
    #: means: scaling a sparse k-mer profile per column would give a k-mer seen
    #: three times the weight of one seen everywhere, while *not* scaling a
    #: handful of physical quantities on incomparable scales leaves the distance
    #: equal to whichever has the largest units. Defaulting this field to a
    #: concrete value made every provider's own default unreachable, because
    #: `compute_block` fills the parameter in before the provider is called.
    normalization: str | None = None
    metric: str = "euclidean"
    representation: str | None = None

    def __post_init__(self):
        path = f"blocks.{self.id}"
        _require_str(f"{path}.id", self.id)
        _require(f"{path}.id", bool(self.id.strip()), "must not be blank")
        _require_str(f"{path}.provider", self.provider)
        _require_mapping(f"{path}.params", self.params)
        _require_bool(f"{path}.fusable", self.fusable)
        if self.normalization is not None:
            _require_choice(f"{path}.normalization", self.normalization, NORMALIZATIONS)
        _require_choice(f"{path}.metric", self.metric, METRICS)
        if self.representation is not None:
            _require_choice(f"{path}.representation", self.representation, REPRESENTATIONS)
        if not self.fusable and not self.not_fusable_reason:
            raise ConfigError(
                f"{path}: fusable is false but no not_fusable_reason was given, and "
                f"{self.id!r} is not one of the signals with a known reason "
                f"({', '.join(sorted(NOT_FUSABLE_REASONS))}). "
                "State the reason: it is shown to anyone who tries to fuse the block, "
                "and it is what stops the flag being removed later as an obstacle."
            )

    @classmethod
    def from_dict(cls, block_id: str, data: Mapping) -> BlockConfig:
        path = f"blocks.{block_id}"
        data = _require_mapping(path, data)
        known = {
            "provider",
            "params",
            "fusable",
            "not_fusable_reason",
            "normalization",
            "metric",
            "representation",
            "fusable_override_reason",
        }
        # Provider-specific keys are common in the wild, so anything not in
        # `known` is folded into params rather than rejected -- but only for a
        # block, where the provider is the authority on its own parameters.
        params = dict(data.get("params", {}) or {})
        for key, value in data.items():
            if key not in known:
                params[key] = value

        provider = _require_str(f"{path}.provider", data.get("provider", block_id))

        # Look the known-overlay-only signals up by *provider first*, then by
        # block id. Keying only on the block id would protect `taxonomy:` and
        # miss `tax:`, `Taxonomy:` and `lineage:` -- i.e. protect exactly the
        # users who already knew. The id lookup is kept, case-folded, as a second
        # chance for a block whose provider name we do not recognise.
        signal = NOT_FUSABLE_PROVIDERS.get(provider.strip().lower())
        if signal is None and block_id.strip().lower() in NOT_FUSABLE_REASONS:
            signal = block_id.strip().lower()
        known_reason = NOT_FUSABLE_REASONS.get(signal) if signal else None

        fusable = data.get("fusable")
        reason = data.get("not_fusable_reason")
        if fusable is None:
            if known_reason:
                fusable, reason = False, reason or known_reason
            else:
                fusable = True
        elif fusable is True and known_reason:
            # An explicit override of a known circularity is allowed, because a
            # maintainer may have a reason we have not thought of -- but it has
            # to be deliberate and written down, not a one-word `true`.
            justification = data.get("fusable_override_reason")
            if not justification:
                raise ConfigError(
                    "\n".join(
                        [
                            f"{path}: fusable is set to true, but this block is a known "
                            f"overlay-only signal (provider {provider!r}).",
                            f"Reason it is normally excluded: {known_reason}",
                            "",
                            "If you have a reason to override this, state it in "
                            f"{path}.fusable_override_reason. It will be recorded in "
                            "the space manifest so that anyone reading the result can "
                            "see the choice was made on purpose.",
                        ]
                    )
                )
            params.setdefault("fusable_override_reason", justification)
        elif fusable is False and not reason:
            reason = known_reason

        return cls(
            id=block_id,
            provider=provider,
            params=params,
            fusable=fusable,
            not_fusable_reason=reason,
            normalization=data.get("normalization"),
            metric=data.get("metric", "euclidean"),
            representation=data.get("representation"),
        )

    def to_spec(self, *, kind: str, **overrides):
        """Build the :class:`spaces.base.BlockSpec` this config describes.

        The bridge between the two descriptions of a block. ``BlockConfig`` is
        what a user writes and can be validated without importing any provider;
        ``BlockSpec`` is what a provider produces and includes facts only the
        provider knows -- whether the block is features or pairwise, and how a
        pairwise block was symmetrized. So ``kind`` comes from the caller, which
        is always the provider.

        Without this, the two would be maintained separately and drift: the
        config layer would be unable to validate its own defaults, and a config
        that passed validation could still be rejected at compute time.
        """
        from spaces.base import BlockSpec

        fields = {
            "id": self.id,
            "kind": kind,
            "fusable": self.fusable,
            "metric": "precomputed" if kind.startswith("pairwise") else self.metric,
            # `to_spec` builds a spec without consulting a provider, so an
            # unset normalization has to become something here. The historical
            # default is the right choice: it is what every block got before
            # the field became optional.
            "normalization": self.normalization or DEFAULT_NORMALIZATION,
            "provider": self.provider,
            "params": dict(self.params),
            "not_fusable_reason": self.not_fusable_reason,
        }
        fields.update(overrides)
        return BlockSpec(**fields)


@dataclass(frozen=True)
class SpaceConfig:
    """One geometry."""

    id: str
    blocks: tuple
    strategy: str = "none"
    weights: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    reducers: tuple = ("pca_umap",)
    #: reducer name -> parameter mapping. Empty means every reducer keeps its
    #: own defaults, which is what every config written before this field did,
    #: so adding it cannot move an existing map.
    reducer_params: dict = field(default_factory=dict)

    def __post_init__(self):
        path = f"spaces.{self.id}"
        _require(f"{path}.blocks", bool(self.blocks), "at least one block is required")
        _require_choice(f"{path}.strategy", self.strategy, STRATEGIES)
        _require(
            f"{path}.blocks",
            len(set(self.blocks)) == len(self.blocks),
            "a block is listed more than once",
        )
        if self.strategy == "none" and len(self.blocks) != 1:
            raise ConfigError(
                f"{path}: strategy 'none' means a single block, but "
                f"{len(self.blocks)} were listed. Either choose a fusion strategy, "
                "or split these into separate co-registered spaces."
            )
        for name, weight in self.weights.items():
            wpath = f"{path}.weights.{name}"
            _require_number(wpath, weight)
            _require(wpath, weight >= 0, f"a negative weight ({weight}) has no meaning")
        unknown = sorted(set(self.weights) - set(self.blocks))
        _require(f"{path}.weights", not unknown, f"weights for blocks not in this space: {unknown}")
        _require(f"{path}.reducers", bool(self.reducers), "at least one reducer is required")
        self._validate_params(path)
        self._validate_reducer_params(path)

    def _validate_params(self, path: str) -> None:
        """Reject a parameter the chosen strategy will not read.

        `params` is a free-form mapping, so a misspelled `iteratons` would be
        carried all the way into the manifest, where it would look configured
        while the run used the default. That is the failure behind FOLLOWUPS #29
        and #32 -- a value written down and honored by nothing -- and the cheap
        place to catch it is here, at parse time, rather than four hours into a
        run or never.
        """
        allowed = STRATEGY_PARAMS.get(self.strategy, frozenset())
        unknown = sorted(set(self.params) - set(allowed))
        if not unknown:
            return
        listed = ", ".join(sorted(allowed)) or "(none)"
        raise ConfigError(
            f"{path}.params: strategy {self.strategy!r} does not read {unknown}. "
            f"Parameters it does read: {listed}."
        )

    def _validate_reducer_params(self, path: str) -> None:
        """Reject a reducer parameter that reducer will not read.

        The same guard as `_validate_params`, for the same reason. UMAP's
        `n_neighbors` is the parameter this field exists for, and it is exactly
        the kind that fails quietly: a misspelling leaves the run on the default
        of 80 while the manifest records the value that was asked for.
        """
        from reduce_space import REDUCER_PIPELINES, params_for

        for reducer, values in self.reducer_params.items():
            rpath = f"{path}.reducer_params.{reducer}"
            _require(
                rpath,
                reducer in self.reducers,
                f"{reducer!r} is not a reducer of this space: {sorted(self.reducers)}",
            )
            _require(
                rpath,
                reducer in REDUCER_PIPELINES,
                f"{reducer!r} is not a known reducer: {sorted(REDUCER_PIPELINES)}",
            )
            values = _require_mapping(rpath, values)
            allowed = params_for(reducer)
            unknown = sorted(set(values) - set(allowed))
            if unknown:
                listed = ", ".join(sorted(allowed)) or "(none)"
                raise ConfigError(
                    f"{rpath}: reducer {reducer!r} does not read {unknown}. "
                    f"Parameters it does read: {listed}."
                )

    @classmethod
    def from_dict(cls, space_id: str, data: Mapping) -> SpaceConfig:
        path = f"spaces.{space_id}"
        data = _require_mapping(path, data)
        known = {"blocks", "strategy", "weights", "params", "reducers", "reducer_params"}
        _reject_unknown_keys(path, data, known)
        blocks = _require_sequence(f"{path}.blocks", data.get("blocks", []))
        reducers = _require_sequence(f"{path}.reducers", data.get("reducers", ["pca_umap"]))
        return cls(
            id=space_id,
            blocks=tuple(_require_str(f"{path}.blocks[{i}]", b) for i, b in enumerate(blocks)),
            strategy=data.get("strategy", "none"),
            weights=dict(data.get("weights", {}) or {}),
            params=dict(data.get("params", {}) or {}),
            reducers=tuple(
                _require_str(f"{path}.reducers[{i}]", r) for i, r in enumerate(reducers)
            ),
            reducer_params={
                str(k): dict(v or {}) for k, v in (data.get("reducer_params", {}) or {}).items()
            },
        )

    @property
    def is_multiblock(self) -> bool:
        return len(self.blocks) > 1

    def weight_for(self, block_id: str) -> float:
        """The weight for a block, defaulting to 1.0 when unspecified.

        Note this is the *pre-normalization* weight. What a block actually
        contributes is its contribution share, which is computed after block
        scale is normalized -- see ADR 0002. A weight of 1.0 alongside another
        block's 1.0 does not imply equal influence until that normalization has
        happened.
        """
        return float(self.weights.get(block_id, 1.0))


@dataclass(frozen=True)
class CoregistrationConfig:
    reference_space: str | None = None
    compare: tuple = ()
    k: int = 10

    def __post_init__(self):
        _require_int("coregistration.k", self.k)
        _require("coregistration.k", self.k > 0, f"must be positive, got {self.k}")

    @classmethod
    def from_dict(cls, data: Mapping | None) -> CoregistrationConfig:
        data = _require_mapping("coregistration", data or {})
        _reject_unknown_keys("coregistration", data, {"reference_space", "compare", "k"})
        compare = _require_sequence("coregistration.compare", data.get("compare", []))
        return cls(
            reference_space=data.get("reference_space"),
            compare=tuple(compare),
            k=data.get("k", 10),
        )


@dataclass(frozen=True)
class EnrichmentConfig:
    """What a cluster is made of, and which columns get asked.

    Column-driven on purpose. Four categories are worth enriching on -- taxon,
    EC number, domain architecture and subcellular localization -- and the
    pipeline fetches two of them:
    `Lineage` and `Pfam`/`InterPro` are in `uniprot_features.tsv`, `ec` and
    `cc_subcellular_location` are never requested (FOLLOWUPS #35). Naming the
    columns rather than the categories means the other two arrive the moment
    the columns do, and means a cohort with its own annotation columns is
    already supported.

    A requested column that is absent is reported by name, never skipped
    silently: "no enrichment for localization" and "localization was never in
    the table" are different facts and only one of them is a finding.
    """

    #: Which column of the cluster table to enrich on. The pipeline's only
    #: clustering today is Leiden over the TM-score matrix, so this describes
    #: the `structure` space rather than the multi-space map -- see ADR 0012.
    cluster_column: str = "LeidenCluster"
    categorical: tuple = ()
    continuous: tuple = ()
    #: Term-encoding overrides, `column -> one of enrichment.ENCODINGS`. Empty
    #: means detect from the column, which is right for every column in
    #: `uniprot_features.tsv` and wrong for a column that is single-valued and
    #: happens to contain a semicolon.
    encodings: dict = field(default_factory=dict)
    #: Terms carried by fewer proteins than this are not tested. A term carried
    #: by one protein cannot reach significance however concentrated it is, and
    #: testing it anyway costs every other term in the family a larger
    #: correction. Dropped terms are counted and reported, not hidden.
    min_term_count: int = 3
    #: Recorded on every row as `significant`, and recorded in the manifest. It
    #: does not filter the table -- the rows that failed it are the evidence
    #: that the ones that passed are not everything.
    fdr: float = 0.05

    def __post_init__(self):
        _require_str("enrichment.cluster_column", self.cluster_column)
        _require(
            "enrichment.cluster_column",
            bool(self.cluster_column.strip()),
            "must not be empty",
        )
        _require_int("enrichment.min_term_count", self.min_term_count)
        _require(
            "enrichment.min_term_count",
            self.min_term_count >= 1,
            f"must be at least 1, got {self.min_term_count}",
        )
        _require_number("enrichment.fdr", self.fdr)
        _require("enrichment.fdr", 0 < self.fdr <= 1, f"must be in (0, 1], got {self.fdr}")
        overlap = sorted(set(self.categorical) & set(self.continuous))
        _require(
            "enrichment",
            not overlap,
            f"{overlap} are listed as both categorical and continuous. A column is "
            "one or the other; testing it both ways would put two incomparable "
            "p-values on the same annotation.",
        )

    @property
    def enabled(self) -> bool:
        return bool(self.categorical or self.continuous)

    @property
    def columns(self) -> tuple:
        return tuple(self.categorical) + tuple(self.continuous)

    def kind_of(self, column: str) -> str:
        return "categorical" if column in self.categorical else "continuous"

    @classmethod
    def from_dict(cls, data: Mapping | None) -> EnrichmentConfig:
        data = _require_mapping("enrichment", data or {})
        known = {
            "cluster_column",
            "categorical",
            "continuous",
            "encodings",
            "min_term_count",
            "fdr",
        }
        _reject_unknown_keys("enrichment", data, known)
        categorical = _require_sequence("enrichment.categorical", data.get("categorical", []) or [])
        continuous = _require_sequence("enrichment.continuous", data.get("continuous", []) or [])
        encodings = _require_mapping("enrichment.encodings", data.get("encodings", {}) or {})
        for column, encoding in encodings.items():
            _require_choice(f"enrichment.encodings.{column}", encoding, ENCODINGS)
            _require(
                f"enrichment.encodings.{column}",
                column in categorical,
                "only a categorical column has a term encoding; "
                f"{column!r} is not listed under enrichment.categorical",
            )
        return cls(
            cluster_column=data.get("cluster_column", "LeidenCluster"),
            categorical=tuple(categorical),
            continuous=tuple(continuous),
            encodings=dict(encodings),
            min_term_count=data.get("min_term_count", 3),
            fdr=data.get("fdr", 0.05),
        )


@dataclass(frozen=True)
class DiagnosticsConfig:
    #: Neighborhood size for trustworthiness and continuity. Clamped down to
    #: the cohort by `diagnostics.embedding.faithfulness` when it does not fit,
    #: with the request kept and reported.
    k: int = 15
    bootstrap_replicates: int = 20
    subsample_fraction: float = 0.8
    leiden_resolution_sweep: tuple = ()
    negative_controls: tuple = ()
    #: ``(censored_space, reference_space)`` -- two spaces built from the SAME
    #: proteins whose only difference is which pairs were measured. Naming them
    #: is what lets the explorer report what censoring did to a map, and it has
    #: to be declared rather than guessed: nothing in a space's own manifest
    #: says "this one is the capped twin of that one", and a convention over
    #: space ids would silently attach the label to whatever happened to match.
    censoring_comparison: tuple = ()

    def __post_init__(self):
        _require_int("diagnostics.k", self.k)
        if self.censoring_comparison:
            _require(
                "diagnostics.censoring_comparison",
                len(self.censoring_comparison) == 2,
                "expects exactly two space ids, [censored_space, reference_space], "
                f"got {list(self.censoring_comparison)!r}",
            )
            for i, space_id in enumerate(self.censoring_comparison):
                _require_str(f"diagnostics.censoring_comparison[{i}]", space_id)
            _require(
                "diagnostics.censoring_comparison",
                self.censoring_comparison[0] != self.censoring_comparison[1],
                "the two spaces must differ; a space compared against itself "
                "measures the reducer, not the censoring",
            )
        _require("diagnostics.k", self.k >= 1, f"must be at least 1, got {self.k}")
        _require_int("diagnostics.bootstrap_replicates", self.bootstrap_replicates)
        _require(
            "diagnostics.bootstrap_replicates",
            self.bootstrap_replicates >= 0,
            "must not be negative",
        )
        _require_number("diagnostics.subsample_fraction", self.subsample_fraction)
        _require(
            "diagnostics.subsample_fraction",
            0 < self.subsample_fraction <= 1,
            f"must be in (0, 1], got {self.subsample_fraction}",
        )
        for value in self.leiden_resolution_sweep:
            _require_number("diagnostics.leiden_resolution_sweep", value)
            _require(
                "diagnostics.leiden_resolution_sweep",
                value > 0,
                f"resolutions must be positive, got {value}",
            )
        _require(
            "diagnostics.leiden_resolution_sweep",
            len(self.leiden_resolution_sweep) != 1,
            "a sweep of one resolution has no adjacent pair to compare, so it "
            "measures nothing. Give at least two, or none at all.",
        )
        _require(
            "diagnostics.leiden_resolution_sweep",
            len(set(self.leiden_resolution_sweep)) == len(self.leiden_resolution_sweep),
            f"resolutions must be distinct, got {list(self.leiden_resolution_sweep)}",
        )
        unknown = [c for c in self.negative_controls if c not in CONTROL_DESCRIPTIONS]
        _require(
            "diagnostics.negative_controls",
            not unknown,
            f"unknown control(s) {unknown}. Known: {sorted(CONTROL_DESCRIPTIONS)}. "
            "A control named here and implemented nowhere is silently skipped, "
            "which is indistinguishable from one that ran and found nothing.",
        )

    @classmethod
    def from_dict(cls, data: Mapping | None) -> DiagnosticsConfig:
        data = _require_mapping("diagnostics", data or {})
        known = {
            "k",
            "bootstrap_replicates",
            "subsample_fraction",
            "leiden_resolution_sweep",
            "negative_controls",
            "censoring_comparison",
        }
        _reject_unknown_keys("diagnostics", data, known)
        return cls(
            k=data.get("k", 15),
            bootstrap_replicates=data.get("bootstrap_replicates", 20),
            subsample_fraction=data.get("subsample_fraction", 0.8),
            leiden_resolution_sweep=tuple(data.get("leiden_resolution_sweep", []) or []),
            negative_controls=tuple(data.get("negative_controls", []) or []),
            censoring_comparison=tuple(data.get("censoring_comparison", []) or []),
        )


@dataclass(frozen=True)
class MultispaceConfig:
    """The whole multi-space configuration, validated as a unit."""

    blocks: dict = field(default_factory=dict)
    spaces: dict = field(default_factory=dict)
    cohort: CohortConfig = field(default_factory=CohortConfig)
    coregistration: CoregistrationConfig = field(default_factory=CoregistrationConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    #: True when the legacy `plotting_modes` key produced these spaces, in which
    #: case the pipeline must keep writing the legacy output paths.
    from_legacy_config: bool = False

    def __post_init__(self):
        self._validate_references()
        self._validate_fusability()
        self._validate_representations()

    # -- cross-cutting validation -----------------------------------------

    def _validate_references(self) -> None:
        for space in self.spaces.values():
            for i, block_id in enumerate(space.blocks):
                if block_id not in self.blocks:
                    known = sorted(self.blocks)
                    raise ConfigError(
                        f"spaces.{space.id}.blocks[{i}]: {block_id!r} is not defined "
                        f"under `blocks`. Defined blocks: {known or '(none)'}"
                    )
        coreg = self.coregistration
        if coreg.reference_space and coreg.reference_space not in self.spaces:
            raise ConfigError(
                f"coregistration.reference_space: {coreg.reference_space!r} is not a "
                f"defined space. Defined spaces: {sorted(self.spaces) or '(none)'}"
            )
        for i, space_id in enumerate(coreg.compare):
            if space_id not in self.spaces:
                raise ConfigError(
                    f"coregistration.compare[{i}]: {space_id!r} is not a defined "
                    f"space. Defined spaces: {sorted(self.spaces) or '(none)'}"
                )
        for i, space_id in enumerate(self.diagnostics.censoring_comparison):
            if space_id not in self.spaces:
                raise ConfigError(
                    f"diagnostics.censoring_comparison[{i}]: {space_id!r} is not a "
                    f"defined space. Defined spaces: {sorted(self.spaces) or '(none)'}"
                )

    def _validate_fusability(self) -> None:
        """ADR 0003, enforced. A non-fusable block may not enter a geometry.

        Applied to any space that combines blocks. A single-block space is not a
        fusion, so an overlay-only signal can still have its own space -- looking
        at a taxonomy-only layout is fine; letting taxonomy move the points in a
        structure map is not.
        """
        for space in self.spaces.values():
            if not space.is_multiblock:
                continue
            for i, block_id in enumerate(space.blocks):
                block = self.blocks[block_id]
                if block.fusable:
                    continue
                raise NotFusableError(
                    "\n".join(
                        [
                            f"spaces.{space.id}.blocks[{i}]: block {block_id!r} cannot "
                            f"be fused into space {space.id!r}.",
                            f"Reason: {block.not_fusable_reason}",
                            "",
                            f"{block_id!r} is available as an overlay on any space, "
                            "which shows the same information without letting it move "
                            "the points. See docs/adr/0003-the-fusable-flag.md.",
                        ]
                    )
                )

    def _validate_representations(self) -> None:
        """`representation: direct` reads the matrix as a matrix (ADR 0007).

        On output written before PR #106 the column order is a permutation of
        the row order, so a direct read returns the wrong cell for almost every
        pair. The loader's assertion is the gate; this check makes the
        requirement explicit at config time rather than at hour four.

        **Alignment is necessary and not sufficient.** It makes the matrix mean
        its labels; it says nothing about whether ``1 - TM`` is a distance that
        can be laid out in a Euclidean space, and on real data it largely is not
        -- 44.6% of the positive spectral mass of the published 160-protein actin
        cohort sits in negative eigenvalues. That second property is measured by
        :func:`diagnostics.metricity.metricity_report` and recorded on the block;
        it is reported and not gated, because the threshold is not knowable from
        one protein family (``docs/FOLLOWUPS.md`` #49).
        """
        for block in self.blocks.values():
            if block.representation != "direct":
                continue
            verified = block.params.get("alignment_verified", False)
            if verified is not False and not isinstance(verified, bool):
                raise ConfigError(
                    f"blocks.{block.id}.alignment_verified: expected true or false, got "
                    f"{type(verified).__name__} ({verified!r}). A quoted string is "
                    "truthy in Python, so 'false' would have switched the gate OFF -- "
                    "which is the opposite of what it says."
                )
            if not verified:
                raise ConfigError(
                    "\n".join(
                        [
                            f"blocks.{block.id}.representation: 'direct' requires "
                            "verified row/column alignment.",
                            "",
                            "'direct' treats the similarity matrix as a matrix, so a "
                            "permuted column order silently returns the wrong cell for "
                            "almost every pair -- measured at 99.92% wrong on a "
                            "pre-#106 production run. 'profile' is safe either way, "
                            "because a consistent permutation of the columns does not "
                            "change distances between rows.",
                            "",
                            "Either use representation: profile, or set "
                            f"blocks.{block.id}.params.alignment_verified: true once "
                            "matrix_io.load_labeled_matrix() accepts your matrix "
                            "without repair.",
                        ]
                    )
                )

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping | None) -> MultispaceConfig:
        data = _require_mapping("<config>", data or {})
        blocks_raw = _require_mapping("blocks", data.get("blocks", {}) or {})
        spaces_raw = _require_mapping("spaces", data.get("spaces", {}) or {})
        return cls(
            blocks={bid: BlockConfig.from_dict(bid, bdata) for bid, bdata in blocks_raw.items()},
            spaces={sid: SpaceConfig.from_dict(sid, sdata) for sid, sdata in spaces_raw.items()},
            cohort=CohortConfig.from_dict(data.get("cohort")),
            coregistration=CoregistrationConfig.from_dict(data.get("coregistration")),
            enrichment=EnrichmentConfig.from_dict(data.get("enrichment")),
            diagnostics=DiagnosticsConfig.from_dict(data.get("diagnostics")),
            from_legacy_config=bool(data.get("from_legacy_config", False)),
        )

    def space_ids(self) -> list:
        return sorted(self.spaces)

    def block_ids(self) -> list:
        return sorted(self.blocks)

    def fusable_blocks(self) -> list:
        return sorted(b for b, cfg in self.blocks.items() if cfg.fusable)

    def overlay_only_blocks(self) -> dict:
        return {b: cfg.not_fusable_reason for b, cfg in self.blocks.items() if not cfg.fusable}


# ---------------------------------------------------------------------------
# legacy bridge
# ---------------------------------------------------------------------------


def _cohort_from_legacy(config: Mapping) -> dict:
    """The `cohort` block, with the top-level `max_structures` folded in.

    Both branches of :func:`from_legacy` need this and an earlier version only
    applied it to one of them, which meant a `cohort:` block in a config with no
    `blocks`/`spaces` keys -- that is, in every existing config -- was silently
    discarded. `selection:` was then a setting that parsed, validated, and did
    nothing. Sharing the code is the fix; keeping it in one place is the point.

    The nested key wins when both are present, so a config can move to `cohort:`
    without deleting the old key first.
    """
    cohort = dict(_require_mapping("cohort", config.get("cohort") or {}))
    if "max_structures" in config and "max_structures" not in cohort:
        cohort["max_structures"] = int(config["max_structures"])
    return cohort


def from_legacy(config: Mapping | None) -> MultispaceConfig:
    """Build a MultispaceConfig from an existing ``config.yml``.

    An existing config has no `blocks` or `spaces` keys. It has
    ``plotting_modes``, which is exactly a list of reducers over the one
    representation the pipeline has always had. So a legacy config becomes a
    single `tmscore` block in a single `structure` space, with `plotting_modes`
    as that space's reducers.

    ``representation: profile`` is the current behavior -- each protein is
    described by its row of the similarity matrix -- and it is deliberately the
    default, because it is invariant to the column permutation that ADR 0007
    describes.

    If the config already carries `blocks`/`spaces`, those win and this is a
    passthrough, so a user can migrate incrementally.
    """
    config = _require_mapping("<config>", config or {})

    if config.get("blocks") or config.get("spaces"):
        merged = dict(config)
        merged["cohort"] = _cohort_from_legacy(config)
        return MultispaceConfig.from_dict(merged)

    modes = config.get("plotting_modes") or ["pca_umap"]
    modes = _require_sequence("plotting_modes", modes)
    for i, mode in enumerate(modes):
        _require_choice(f"plotting_modes[{i}]", mode, LEGACY_PLOTTING_MODES)

    return MultispaceConfig.from_dict(
        {
            "blocks": {
                LEGACY_BLOCK_ID: {
                    "provider": "tmscore",
                    "representation": "profile",
                    "normalization": "unit_mean_distance",
                }
            },
            "spaces": {
                LEGACY_SPACE_ID: {
                    "blocks": [LEGACY_BLOCK_ID],
                    "strategy": "none",
                    "reducers": list(modes),
                }
            },
            "cohort": _cohort_from_legacy(config),
            # Enrichment is carried through the legacy branch as well, because
            # it needs a cluster table and an annotation table and a legacy run
            # has both. Dropping it here would silently ignore an `enrichment:`
            # key in a plain cluster-mode config, which is the exact failure
            # `_reject_unknown_keys` exists to prevent one level up.
            "enrichment": config.get("enrichment") or {},
            # And diagnostics, for the same reason. This key was dropped here
            # until group 8b, which is the failure the comment above describes
            # happening one line below itself: a legacy config could set
            # `diagnostics.k` and be ignored, or misspell it and not be told.
            # Found by the test FOLLOWUPS #36 asked for, not by a reader.
            "diagnostics": config.get("diagnostics") or {},
            "from_legacy_config": True,
        }
    )
