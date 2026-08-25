# ADR 0006 — How optional and licensed dependencies stay out of CI

Status: accepted
Date: 2026-08-16

## Context

The representation blocks this project adds have wildly different dependency
costs. `threedi` needs foldseek, which is already a dependency. `biophys` needs
Biopython. `plm` needs torch and a multi-gigabyte checkpoint. `localization`
needs DeepLoc 2.1 and SignalP 6.0, which are academically licensed and cannot be
redistributed. MEROPS has its own terms. ESM-C requires accepting a licence on
HuggingFace.

**A maintainer cannot accept a PR that makes their pipeline depend on artifacts
they cannot redistribute, cannot test against, and cannot fix when they break.**
This is the single most likely reason for this PR to be rejected, so the answer
has to be structural rather than a promise in a README.

The repo has also already been burned by dependency drift in a way that
constrains the solution:

- `envs/analysis.yml` pinned `numpy=1.23.5` but not its transitive dependents,
  so a fresh solve installed a numpy-2-built matplotlib beside it — an ABI break
  the solver accepts silently and that fails only at import. The fix is now
  in-file: `matplotlib=3.7.1` with a comment explaining exactly this.
- `setuptools >= 81` removed `pkg_resources`, which `umap-learn 0.5.3` imports at
  module scope. Also now pinned in-file with a comment.
- The CI conda cache key hashes the env files, so CI never re-solves while they
  are unchanged — which is precisely why the drift went undetected for so long.
  It was `hashFiles('envs/*.yml')` when this record was written and is now the
  six files a `conda:` directive actually names, because hashing every env file
  meant editing an env no rule solves evicted a multi-gigabyte cache and rebuilt
  every pipeline environment. The narrowing does not weaken the point above: at
  HEAD there are **nine** env files and the **three** excluded ones —
  `cartography_pub`, `cartography_test`, `cartography_tidy` — are not solved by
  snakemake at all. It was ten and four when this was written; the
  `cartography_dev.yml` deletion recorded below is what changed the count, and
  the same stale "all ten" is copied into `multispace.yml:98` and `:308`.
- `mamba` 2.x removed the `mamba env create` CLI that snakemake 7.25.3 invokes,
  so any new env that leaves mamba unpinned breaks `--conda-frontend mamba`.

## Decision

**Four rules, enforced in code and CI rather than documented.**

> **What exists today.** This record is written in the present indicative, and
> the blocks it is chiefly about — `plm`, `function`, `localization` — belong to
> Phases 8 and 9 and are not in this PR. Rules 1 to 4 are all in force for what
> *is* here: `envs/` is untouched by this branch **except that
> `cartography_dev.yml` was deleted** — a drifted duplicate of `analysis.yml`
> that no rule solved, pinning sklearn 1.3.2 against the determinism
> guarantee's 1.2.2; see FOLLOWUPS #2. Rule 1 below is about what enters an env
> file, and nothing entered one. Every provider implements
> `is_available`. **No optional dependency is load-bearing in the driver
> environment** — the `end-to-end-with-no-optional-dependencies` job in
> `.github/workflows/multispace.yml` guards that env (`:72-85`) and runs the demo
> through it (`:114`), and
> `ProteinCartography/tests/test_snakefile_parses_without_optional_dependencies.py`
> pins the same property in a subprocess. That test exists because the property
> was not true when this was written: `Snakefile:167` reached `bioservices`
> through `config_utils`, so `snakemake -n` could not parse the Snakefile in any
> environment built from `envs/cartography_test.yml`. It says nothing about the
> blocks: `compute_block` and `reduce_space` run under `envs/analysis.yml`
> (`Snakefile:932`, `:966`), which ships scikit-learn, umap-learn and scanpy.
> Items marked **(deferred)** below arrive
> with the phase that needs them and do not exist yet; a reviewer following one
> of them today reaches nothing.

**1. Nothing heavy or licensed enters an existing env file.**
`envs/cartography_tidy.yml` and `envs/analysis.yml` are untouched. New capability
gets new env files — `envs/embeddings.yml`, `envs/function.yml`,
`envs/structure_tools.yml` **(deferred; none of the three exists, because none
of the blocks needing them is in this PR)**. Each pins `mamba=1.4.2`, and none pins `python=`
without timing evidence — an earlier revision that added `python=3.9.16` to
three envs took `plotting.yml` from ~74 s to over 12 minutes without finishing.

**2. Every provider implements `is_available() -> tuple[bool, str]`.**
It reports whether both the package *and* its weights are present, and the
string explains what is missing and how to get it. `compute_block` skips an
unavailable **block** with a log line on stderr and exits 0
(`compute_block.py:105-108`); the Snakefile itself never consults availability
and does not import the registry at all, which `spaces/registry.py:196-199`
states outright. A missing optional dependency is a reduced result, never an
error.

**3. The shipped multispace config names only free, ungated blocks.** `tmscore`,
`threedi`, `biophys`, `domains`, in `demo/multispace/config.yml`; the default
`config.yml` names no `blocks:` and no `spaces:` at all, so the legacy pipeline
carries none of this either way. The framework and all four must be fully
functional and tested with zero optional dependencies installed. Today only
`tmscore`'s `profile` path has a bare-environment test
(`tests/test_optional_dependencies.py:134`); the rules themselves run under
`envs/analysis.yml`, so the rest of the rule is a requirement, not a measured
fact. Where a gated model has a free counterpart, the free one
is the default — `plm` defaults to ESM-2 650M (open
weights on the HF hub) rather than ESM-C (licence acceptance required), so the
PLM code path is CI-testable upstream without anyone accepting anything.

**4. CI proves rule 3 rather than asserting it.** The standing test job installs
zero optional dependencies and runs the pipeline end to end. If any optional
dependency becomes load-bearing, that job fails. Network and GPU jobs are
manually triggered or scheduled, never on the PR path.

Two supporting mechanisms:

- **An import smoke test per environment (deferred).** Building an environment
  does not prove it imports — that is the exact lesson of the numpy/matplotlib
  ABI break. Each env gets a job that creates it and imports its top-level
  packages.
- **A scheduled fresh-solve job (deferred; no workflow carries a `schedule:`).**
  Because the cache key is a hash of the env files, unchanged ones are never
  re-solved and drift is invisible. A weekly job that bypasses the cache is the only thing that surfaces
  it.

**Licensed blocks are designed to be droppable.** Each is one config entry plus
one provider file, with no other module importing it. If review pressure appears,
they drop first and the cost is deleting a config block — no code changes
elsewhere.

**(Deferred, with the first licensed block.)** `docs/MODELS.md` will record
every optional model with its licence, acquisition steps, hardware needs, and
what degrades without it, and `make fetch-models` will download and hash
checkpoints into a configurable cache dir, with the hashes going in the space
manifest. Neither exists: this PR ships no model-backed block, so there is
nothing yet to record or fetch.

## Consequences

- Upstream CI stays green with none of this installed, and there is a test that
  fails if that stops being true.
- A reviewer's question "does this make my pipeline depend on DeepLoc?" has a
  one-word answer backed by a CI job.
- Users who want the heavy blocks pay a real setup cost: create an env, accept a
  licence, fetch weights. This is the correct place for that cost.
- Provider code is slightly more verbose — every provider carries an
  availability check even when it is trivially `True`.
- Blocks that are skipped produce absent columns, so downstream code must
  tolerate missing spaces. Nothing enforces that today: co-registration works
  from the explicit `coregistration.compare` list in the config
  (`config_schema.py:613`, each entry validated against the defined spaces at
  `:855`), not from whatever spaces exist.

## Alternatives rejected

**Optional extras in `pyproject.toml` (`pip install .[plm]`).** Rejected as
insufficient on its own: the pipeline is conda-driven per-rule via snakemake
`conda:` directives, and pip extras do not express "this rule needs torch and
that one does not". Extras may still be added later as a convenience, but they
cannot be the mechanism.

**Vendor the licensed models.** Not permitted by their licences, and it would
make the repo enormous. Non-starter.

**Put everything in one big environment and let users not use what they don't
want.** Rejected: it makes every user pay the solve time and disk cost of torch,
and it makes CI depend on gated artifacts. This is the failure mode the ADR
exists to prevent.

**Make blocks fail loudly when dependencies are missing.** Rejected: it converts
an optional feature into a required one in practice. A missing optional block
must degrade the result, not break the run.
