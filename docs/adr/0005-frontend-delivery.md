# ADR 0005 — Frontend delivery: one static self-contained HTML file

Status: accepted
Date: 2026-08-16

## Context

Co-registration's primary output is *several maps that must be looked at
together*, with linked selection, because the discovery signal is disagreement
between spaces. That is a genuine interactive-UI requirement, not a chart.

The existing pipeline already ships interactive HTML via
`plot_interactive.py` → Plotly `fig.write_html(...)`, one file per plotting mode,
landing in `final_results/`. That is how ProteinCartography artifacts are
distributed today: attached to a pub, dropped in a Slack thread, archived beside
the data.

The tempting alternative is a small served app — FastAPI plus a JS frontend —
which would allow live re-embedding, server-side computation, and a nicer
development experience.

## Decision

**A single static, self-contained HTML file. All data embedded. No server, no
CDN, no build step.**

`final_results/{analysis_name}_explorer.html`, built with vanilla JS and Plotly
(already a dependency via `envs/plotting.yml`). No `localStorage`, no external
fetches, no npm.

**The optional FastAPI server stays out of this PR entirely.** It has the highest
maintenance burden and the weakest claim to belong upstream. It lives in the
fork.

The decisive argument is not technical: **a served app is a runtime service the
maintainer would have to own.** Accepting it means accepting responsibility for
its dependencies, its security surface, its deployment story, and its bit-rot.
A static file has none of those, and it is the only form that survives archival —
a pub from three years ago should still open.

Contents, in priority order:

1. **Multi-panel layout** — co-registered spaces over one `protid` index, one
   panel per space the config declares, with no cap in the code
   (`explorer/payload.py:990`). The shipped demo declares seven
   (`demo/multispace/config.yml`), not the 2–4 anticipated here.
2. **Linked selection.** Built first; everything else is secondary. Selecting
   points in one panel highlights the same proteins in all others. This is the
   feature that makes co-registration legible.
3. **Global overlay selector.** Built, but not from `plotting_rules`: the
   vocabulary is whatever columns the run's aggregated feature table already
   carries, kept only where a column is usable as a colour
   (`explorer/payload.py:682-700`). `plotting_rules` is a code-level dict in
   `plot_cluster_distributions.py`, and nothing under `explorer/` imports from
   it — which is what the last paragraph of "Alternatives rejected" already says.
   The two statements had been contradicting each other.
4. **Disagreement mode** — color by cross-space neighborhood Jaccard. One click,
   not buried in a menu.
5. **Diagnostics overlays** in the same dropdown but **visually grouped
   separately**, so a QC metric is never mistaken for a biological finding.
6. **Contribution shares printed on the panel** for every fused space, both the
   asked-for and the realized split, which differ (ADR 0002). The **preset
   switcher** they were to sit beside is **not built**: it needs precomputed
   alternative weightings, and this PR fuses each pair once. When it arrives it
   goes here.
7. **Cluster inspector** — FDR-corrected enrichment, cross-cluster similarity,
   stability distribution. **Not built.** The enrichment table is computed and
   written to `enrichment/cluster_enrichment.tsv`; nothing in the explorer reads
   it yet.
8. **Provenance footer** — manifest, versions, dates, seeds, N, cohort rule.
   Always visible, never collapsed. **Superseded on one item: the date was
   deliberately dropped.** Two runs of the same inputs must produce the same
   bytes, and a generation timestamp would be the one field guaranteeing they
   never do (`explorer/payload.py:1746-1753`). The page says so in the footer,
   where the date would otherwise have been
   (`explorer/template.py:2961-2962`).

**Be honest about the one real constraint.** Live weight-slider re-embedding
requires recomputing a distance matrix and a projection, which is not feasible
client-side at realistic N. The mixer switches between **precomputed named
presets**. A slider that silently snaps to the nearest preset is worse than
labeled buttons, because it implies a continuity that does not exist. **None are
shipped in this PR** — "six of them shipped" described an intended end state and
was read, reasonably, as a description of the page.

**`plot_interactive.py` keeps working and keeps emitting its existing
filenames.** The explorer is additive.

## Consequences

- File size grows with N, since all coordinates and features are embedded. At
  N=5000 with four spaces and the standard feature set this is a few MB of JSON
  — large for a web page, unremarkable for a data artifact. Above N≈20,000 it
  would need attention; that is beyond the pipeline's compute ceiling anyway
  (ADR 0004).
- No server-side computation means everything shown must be precomputed. This is
  a real constraint on features and it is why presets exist.
- Plotly is already a dependency, so this adds no new runtime requirement.
- Debugging a single generated HTML file is less pleasant than a dev server with
  hot reload. Accepted, and mitigated by keeping the JS small and generating it
  from templates rather than string concatenation.
- The file works offline, opens from `file://`, survives being emailed, and can
  be committed beside a pub. These are the properties that matter for this
  audience.

## Alternatives rejected

**FastAPI + served frontend.** Rejected for this PR — a runtime service the
maintainer must own, for a benefit (live re-embedding) that is not achievable at
realistic N anyway. Kept in the fork for local exploratory use.

**Jupyter notebook / voila.** Rejected: requires a running kernel, does not
archive, and does not survive being sent to a collaborator.

**Extend `plot_interactive.py` to render all panels.** Rejected: it is over
1,100 lines — 1,105 when this was written, 1,144 at HEAD, since #110 and this
branch both touched it — has no rule-injection mechanism, and derives its axes
positionally from `df.columns[1]` and `[2]` (`plot_interactive.py:877-878`). Extending it would mean a large diff in an existing
file to serve a use case it was not shaped for. The explorer shares no code
with it: it defines its own overlay vocabulary in `explorer/`, and
`generate_plotting_rules` is imported nowhere under `ProteinCartography/explorer/`
or in `build_explorer.py`. This paragraph previously asserted that import. It
never existed, and no reviewer or test could have caught the claim, because an
ADR is prose. The palette is deliberately NOT unified now: doing it would
recolour a page whose red-ring-over-overlay-fill semantics were set by a manual
browser pass that nothing automated re-verifies, and it would be a drive-by
change under this branch's additive-only rule. If a reviewer wants one palette,
that is its own ticket with its own browser pass.

**A React/Svelte app bundled to a single file.** Rejected: adds a build step and
a node toolchain to a conda/snakemake repo. The interaction requirements —
linked selection across a few Plotly panels and a dropdown — do not justify a
framework.

**Bundle a CDN link instead of inlining Plotly.** Rejected: it breaks offline
use and breaks the moment the CDN version changes or disappears, which defeats
the archival property that motivated the whole decision.
