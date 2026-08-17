# ADR 0001 — Block / Space / View, and why the seam is at the representation

Status: accepted
Date: 2026-08-16

## Context

ProteinCartography today builds one map from one representation: an
all-versus-all Foldseek TM-score matrix → PCA-30 → UMAP or t-SNE, with metadata
joined at the end as color overlays. It answers one question well — *does global
fold structure organize this family?*

It cannot answer several questions people keep asking of it: which proteins are
structurally alike but sequence-divergent, which subfamilies split on surface
chemistry or predicted function rather than fold, how much of a layout is an
artifact of Foldseek's per-query hit cap, and how far a lineage travelled through
structure space relative to its branch length.

Every one of those needs *more than one representation over the same proteins*.
So the question is where to put the extensibility seam.

Three candidate seams exist in the current code:

1. **At the reducer** — keep one input matrix, add more projections of it.
2. **At the representation** — allow several independent per-protein
   representations over one `protid` index, each with its own metric.
3. **At the plot** — keep one geometry, add overlays.

The current pipeline already does (1) — `plotting_modes: [pca_tsne, pca_umap]`
— and (3), via `plotting_rules`. Neither answers any of the questions above,
because both are downstream of a single fixed geometry.

## Decision

**The seam is at the representation.** Three concepts, with a strict
responsibility boundary:

**Block** — a per-protein representation. Either a feature matrix
`X ∈ ℝ^(N×D)` or a precomputed pairwise distance. Declares its metric, its
normalization, and whether it may enter a geometry (`fusable`, ADR 0003).

**Space** — a geometry. Blocks + a combination strategy + a metric + a reducer.
Produces coordinates.

**View** — a rendering. **Never changes geometry.**

```python
@dataclass(frozen=True)
class BlockSpec:
    id: str
    kind: Literal["features", "pairwise"]
    fusable: bool
    metric: str            # euclidean | cosine | precomputed | jaccard
    normalization: str     # none | zscore_within | unit_mean_distance
    provider: str
    params: dict
    not_fusable_reason: str | None = None

@dataclass(frozen=True)
class BlockResult:
    spec: BlockSpec
    protids: list[str]     # canonical order — identity lives here, not in position
    features: np.ndarray | None
    distances: np.ndarray | None   # condensed
    mask: np.ndarray | None        # censoring — ADR 0009
    manifest: dict                 # provenance
```

Providers are discovered through `pyproject.toml` entry points, so **a third
party can add a representation without editing any existing file**.

Two consequences of the split are load-bearing and worth stating explicitly:

- **Overlay never moves points; fusion always does.** The distinction is visible
  in the API, the config schema, the output filenames, and the UI. A `View` has
  no way to alter coordinates, because it is handed coordinates rather than
  blocks.
- **Co-registration is the default product, not fusion.** Several spaces computed
  independently over one `protid` index, displayed side by side with linked
  selection. The primary discovery signal is *disagreement between spaces*.
  Fusion is an explicitly invoked analysis (ADR 0002).

**The canonical `protid` index is the contract between blocks.**
`index.align(block_result, index)` **raises** on a missing protid rather than
reindexing to NaN. Silent reindexing is the single most likely source of a
subtle wrong answer in this design, and pandas will do it by default, so the
guard is explicit.

**`representation: direct` requires an explicit symmetrization decision.**
Measured on production output, the TM matrix is genuinely asymmetric even where
both directions were reported: only 40.2% of both-reported cell pairs are exactly
equal, median absolute difference 0.0014, max 0.6711. This is correct behavior —
TM-score is length-normalized per query, so `TM(a→b) ≠ TM(b→a)` whenever lengths
differ. Any provider offering `direct` must record which of min / max / mean /
kept-asymmetric it applied, in the manifest. It may not assume symmetry.

## Consequences

- The existing TM path becomes one block among several, ported behind a parity
  test that proves the port changed nothing (commit group 5).
- `dim_reduction.py` becomes a thin shim over `spaces/reducers/`, preserving its
  CLI exactly.
- New capability is new files in new directories. The existing-file diff is
  concentrated in the Snakefile.
- Every space carries a manifest sufficient to recompute it exactly: block
  versions, weights, normalization, metric, reducer params, seeds, input hashes.
- Cost: three concepts where there was one script. Justified only because the
  questions in the Context section each require more than one representation;
  if only one were ever needed, this would be over-engineering.

## Alternatives rejected

**Seam at the reducer.** Rejected: it multiplies projections of the same
geometry. Every question in the Context section requires different *input*, not
a different projection of the same input. This is what the pipeline already does
and it is why those questions are unanswerable today.

**Seam at the plot (overlays only).** Rejected for the same reason, and worse:
an overlay of a sequence-derived quantity on a structure geometry invites the
reader to interpret proximity in that overlay's terms, when proximity was
computed from structure alone. Overlays are necessary but not sufficient.

**One fused geometry by default.** Rejected as scientifically backwards. Fusing
first destroys the disagreement signal that is the most interesting output.
Co-registration preserves it and fusion remains available when a single geometry
is genuinely wanted (ADR 0002).

**Subclassing a `Block` base class instead of a Protocol + dataclasses.**
Rejected: providers live in separate optional packages with separate
dependencies (ADR 0006). A Protocol lets a provider be written without importing
this package's class hierarchy, and keeps `is_available()` checkable before any
heavy import happens.
