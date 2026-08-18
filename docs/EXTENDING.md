# Adding a block without editing this repository

A **block** is one measurement of the proteins: an `(N, D)` feature matrix, or a
precomputed `(N, N)` distance. `tmscore`, `threedi`, `biophys` and `domains` are
the four that ship. Adding a fifth requires no change to any file here.

That claim is the whole point of the registry, so it is worth stating precisely:
**a third-party package declares an entry point, and `registry.py` finds it.**
Nothing in this repo is edited, nothing is imported by name, and if your
dependency is missing the pipeline logs one line and carries on without your
block rather than failing (ADR 0006).

`docs/ARCHITECTURE.md` sketches this in eight lines. This document is the
worked version, with the parts that are easy to get wrong called out.

---

## 1. Declare the entry point

```toml
# in your own package's pyproject.toml
[project.entry-points."proteincartography.blocks"]
hydropathy_windows = "mypkg.blocks:HydropathyWindowsProvider"
```

The group name is `proteincartography.blocks`, exported as
`spaces.registry.BLOCK_GROUP`. **It is the only group that is resolved.**
`REDUCER_GROUP` and `FUSION_GROUP` are defined beside it and reserved for later;
nothing reads them today, so a reducer or fusion entry point is silently
ignored. Both are closed sets — see §7.

The entry-point *name* is what a config's `provider:` field names. Keep it
stable: it goes into every manifest.

---

## 2. Write the provider

A provider is a plain object with four members. There is no base class to
inherit and no import from this package required — it is a structural protocol,
so your package need not depend on ProteinCartography at all to satisfy it.

```python
class HydropathyWindowsProvider:
    """Mean Kyte-Doolittle hydropathy in W windows along each sequence."""

    #: Validates and normalizes the block's `params` from the config.
    #: Declarative only: see the note under `spec_schema` below. Call it
    #: yourself, first thing in `compute`, as the built-in providers do.
    spec_schema = staticmethod(validate_params)

    #: Bump when the output's *meaning* changes. Recorded on every block, and
    #: **not yet part of the cache key** — `Manifest.cache_key` excludes
    #: `derived`, which is where the resolved spec lands, so a bump does not
    #: currently invalidate anything. `docs/FOLLOWUPS.md` #46.
    version = "1"

    def is_available(self) -> tuple[bool, str]:
        try:
            import mydep  # noqa: F401
        except ImportError as error:
            return False, (
                f"hydropathy_windows needs mydep ({error}). "
                "Install it with `pip install mydep`."
            )
        return True, ""

    def compute(self, ctx, params: dict) -> BlockResult:
        ...
```

### `spec_schema`

A callable taking the raw `params` dict and returning a validated one, raising
on anything it does not recognise. **Reject unknown keys.**

**The framework does not call it.** It is declared by all four built-in
providers and read by nothing outside the tests: validation happens because each
provider calls its own `validate_params` at the top of `compute()`, which is
*inside* the snakemake rule and therefore after the run has started. The reason
is deliberate — `config_schema.py` validates a config without importing any
provider, so it cannot reach a provider's schema at parse time — but the effect
is that a third-party provider which declares `spec_schema` and trusts it to run
gets no parameter validation at all. Call it yourself. A misspelled
parameter that is silently ignored is indistinguishable from one that is read
and does nothing, which is the most persistent defect shape in this codebase —
`spec.metric` and `spec.normalization` are each recorded on every block and
consulted by nothing (`docs/FOLLOWUPS.md` #29, #32).

### `is_available`

Returns `(bool, str)`. The string must say *what* is missing and *how to get
it*, because it is shown to the user verbatim. Check for weights and data files,
not only for importable packages — a model whose checkpoint has not been
downloaded is not available even though its library imports fine.

**A missing optional dependency is a reduced result, never an error** (ADR 0006
rule 2). The Snakefile skips an unavailable block with a log line.

### `compute`

Returns a `BlockResult`:

```python
BlockResult(
    spec=BlockSpec(
        id=params["id"],
        kind="features",              # or "pairwise"
        fusable=True,                 # see §4
        metric="euclidean",
        normalization="unit_mean_distance",
        provider="hydropathy_windows",
        version=self.version,
    ),
    protids=protids,                  # the canonical order — identity lives here
    features=matrix,                  # (len(protids), D)
    channels={},                      # optional, e.g. {"censored": mask}
)
```

`ctx` gives you the run: `ctx.path(*parts)` resolves inside the output
directory, and `ctx.extras` carries paths the Snakefile resolved for you (the
features table's location depends on search vs cluster mode, so it arrives here
rather than in `params` — a machine-specific path in `params` would put a
machine-specific value in the cache key).

---

## 3. The five rules that will bite you

These are not style preferences. Each one is a defect this project has already
shipped and fixed.

**1. Identity lives in labels, never in position.** `protids` is the canonical
order and every array is interpreted against it. Do not assume your input file's
row order matches anyone else's. On a real 2530-protein matrix, only **2 of 2530
columns** sat in their row position — the header was written from a Python set,
whose iteration order is salted per process. Reading it positionally gets almost
every cell wrong. Load labeled matrices through `matrix_io`, which refuses a
misaligned one (ADR 0007).

**2. Say whether your zeros are measurements.** If your block can produce a cell
that means *not measured* rather than *measured as zero*, carry a boolean mask
in `channels["censored"]`. 60.5% of a production TM-score matrix is fill, and
treating it as dissimilarity is a category error (ADR 0009).

**3. Do not put length in the geometry by accident.** Any descriptor that grows
with sequence length — molecular weight, residue counts, total charge — makes
protein length a principal axis of a biology map. `biophys` warns loudly when
one is requested and records `length_proportional_descriptors` in the manifest.
Prefer intensive quantities (per-residue, fractions, means).

**4. Cohort-scoped vocabularies are not comparable across runs.** If your block
builds its columns from the observed proteins — a k-mer set, a domain
vocabulary — then two runs over different cohorts produce matrices whose columns
mean different things. That is fine for one run and must be recorded in the
manifest. Accept an explicit `vocabulary=` so a caller can pin one
(FOLLOWUPS #28, #31).

**5. A value written to a manifest is not thereby honored.** If you add a field,
add the test that proves something reads it. Three separate fields in this
codebase were recorded, validated, looked like evidence, and were consulted by
nothing.

---

## 4. When your block must *not* enter a geometry

Some signals belong on a map as an overlay and never as an axis. Set
`fusable=False` and give a `not_fusable_reason`, which is shown to the user
verbatim when they try to fuse it.

The existing cases and their reasons are in
`config_schema.NOT_FUSABLE_REASONS`: taxonomy (fusing it makes every
taxon-specific cluster claim circular), phylogeny (patristic distance is derived
from the same sequences), pLDDT and prediction confidence (both track length),
censoring rate (a property of how well a protein was measured, not of the
protein), and `struclusters` (a structurally-gated graph clustered on
amino-acid identity, so using it inside a geometry later contrasted against
sequence space is partly circular).

The rule is keyed on the **provider name**, not the block id, deliberately: the
block id is free text the user chooses, so a table keyed on it would protect
`taxonomy:` and miss `tax:`, `Taxonomy:` and `lineage:` — that is, it would
protect exactly the users who already knew.

A single-block space is not a fusion, so an overlay-only signal can still have
its own space. Looking at a taxonomy-only layout is fine; letting taxonomy move
the points in a structure map is not (ADR 0003).

---

## 5. Use it

```yaml
blocks:
  hydropathy:
    provider: hydropathy_windows
    params:
      windows: 8

spaces:
  chemistry:
    blocks: [hydropathy, biophys]
    strategy: late
    reducers: [pca_umap]
```

`strategy: late` normalizes each block to unit mean distance before combining,
which is what makes two blocks at different scales comparable — `wide` at scale
10 and `narrow` at scale 0.01 must not be combined raw (ADR 0002, ADR 0013).

---

## 6. Test it the way this repo tests blocks

The pattern is not optional ceremony; it is what caught the defects above.

**Build a fixture with a known answer before writing the statistic.** Not a
fixture that checks shapes — one where the right answer is fixed by
construction, so a structurally perfect but meaningless result fails.
`tests/fusion_cohort.py` plants two exactly crossed partitions;
`tests/embedding_cohort.py` plants a 2×2 of faithfulness that no single
statistic can fake. This has changed the design before the code existed four
times running.

**Cross-check any reimplemented statistic against a reference**, behind
`pytest.importorskip`, and then *actually run it somewhere it does not skip* — a
gated test can be written, be correct, and never execute.

**Run it end to end, at the demo's scale and not only the fixture's.** Seven
defects in this work were reachable only that way, and several were pure
small-N: a default `k` of 15 against an 11-protein demo, a neighbourhood that is
the entire cohort, a constant block whose distances cancel to `4e-08` at one
cohort size and exactly `0.0` at another.

**Confirm your block imports with nothing installed.** Add it to
`tests/test_optional_dependencies.py`. Heavy imports go inside the function that
uses them, so `is_available()` can be consulted before anything is loaded.

---

## 7. Reducers and fusion strategies

Neither is extensible by entry point today, despite `REDUCER_GROUP` and
`FUSION_GROUP` existing. Reducers are resolved from the closed
`reduce_space.REDUCER_PIPELINES` dict; adding one means editing that dict. What
the group would supply, if it is ever wired, is a callable returning a
`ReducerResult`
(coordinates, protids, column names, and `params_used` *after* clamping — a run
at N=4 that asked for 80 neighbours and got 3 must say so, or two runs with
identical configs and different N look identical in provenance).

Fusion is deliberately not extensible, rather than merely not wired. The four strategies
are a closed set that the config validator has to enumerate anyway, and each
one's parameters are declared in `fusion.STRATEGY_PARAMS` so an unknown
parameter is rejected rather than ignored (ADR 0013 §6).
