# ADR 0010 — Config validation without pydantic

Status: accepted
Date: 2026-08-16

## Context

The config schema needs real validation: type checking, required keys, value
constraints, and the `fusable: false` rejection that ADR 0003 depends on. The
obvious tool is pydantic v2, and that was the original plan.

Three facts make it the wrong choice here.

**1. The validator has to run inside the snakemake driver environment.**
`config_utils` is imported at the top of the `Snakefile`
(`Snakefile:4`), and the Snakefile is evaluated by snakemake itself. ADR 0003
requires the `fusable` rejection to fire at config-load time rather than after
an expensive computation, which means the validator runs wherever snakemake
runs — `envs/cartography_tidy.yml`.

**2. That environment is closed.** Adding dependencies to
`envs/cartography_tidy.yml` is explicitly off the table for this work. pydantic
is not currently declared in any environment file in the repo.

**3. pydantic v2 is a compiled dependency.** `pydantic-core` is a Rust extension
module, so it is ABI-coupled to its Python build and adds a wheel-availability
constraint on every platform the pipeline runs on, including the arm64 Macs the
README documents. This repo has already lost time to exactly this class of
problem twice: a numpy-2-built matplotlib installed silently beside
`numpy=1.23.5` and failing only at import, and `setuptools>=81` removing
`pkg_resources` out from under `umap-learn 0.5.3`. Both are now pinned in
`envs/analysis.yml` with comments explaining why.

Adding a compiled dependency to the one environment that must always work, in a
PR whose central promise is that it changes nothing by default, is a poor trade
for validation ergonomics.

## Decision

**Config models are frozen `dataclasses` with explicit validation in
`__post_init__`.** No third-party dependency. Standard library only.

```python
@dataclass(frozen=True)
class BlockConfig:
    id: str
    provider: str
    params: dict = field(default_factory=dict)
    fusable: bool | None = None      # None = take the provider's default
    normalization: str = "unit_mean_distance"

    def __post_init__(self):
        _require_identifier("block id", self.id)
        _require_choice("normalization", self.normalization, NORMALIZATIONS)
```

Supporting helpers give pydantic-shaped error messages — the field path, the
offending value, and the allowed set — because the error text is the part of
validation users actually interact with:

```
spaces.multiview.blocks[2]: block 'taxonomy' cannot be fused.
Reason: fusing taxonomy makes every taxon-specific cluster claim circular ...
```

**Provider parameter schemas stay pluggable.** `BlockProvider.spec_schema` is a
callable that validates and normalizes a params dict and raises
`ConfigError` on bad input. A provider living in a third-party package that
already depends on pydantic is free to implement `spec_schema` with a pydantic
model — the contract is the callable, not the library. This keeps the door open
without making the core depend on it.

**If pydantic is ever added to the driver environment for another reason**, the
dataclasses can be swapped for `pydantic.dataclasses` with essentially no call-site
change. The decision is reversible.

## Consequences

- Roughly 150 lines of validation helpers that pydantic would have provided.
  This is the real cost and it is not zero.
- Error messages are hand-written, which is more work but produces better
  messages for the cases that matter — the `fusable` rejection needs to explain
  circularity, and no schema library would generate that text.
- No JSON Schema export for free. Nothing currently needs one.
- No coercion. A config that says `max_structures: "5000"` is rejected rather
  than silently coerced, which is the behavior we want for a scientific
  pipeline: a string where an int belongs is a mistake worth surfacing.
- Zero new dependencies in any environment, so ADR 0006's "CI passes with no
  optional dependencies" claim holds for the config layer trivially.

## Alternatives rejected

**Use pydantic and add it to `envs/cartography_tidy.yml`.** Rejected: the
environment is closed for this work, and a compiled dependency in the driver env
is the highest-blast-radius place to add one.

**Use pydantic as an optional import with a dataclass fallback.** Rejected as the
worst of both — two validation code paths that can disagree, with the divergence
surfacing only on machines that have one and not the other. Validation
inconsistency is a bug class that is very hard to reproduce.

**Use `jsonschema`.** Also a dependency, pure-Python but still absent from the
driver env, and it produces worse error messages for nested config than
hand-written checks do.

**Skip validation; let it fail at use.** Rejected outright. ADR 0003's entire
enforcement mechanism is the validator, and "the run failed four hours in
because taxonomy was fused" is exactly the outcome that invariant exists to
prevent.

**Use `attrs`.** Same dependency objection as the others, for a smaller benefit
over stdlib dataclasses than pydantic would have offered.
