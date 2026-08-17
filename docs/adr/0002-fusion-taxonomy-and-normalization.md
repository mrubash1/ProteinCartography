# ADR 0002 — Fusion taxonomy and the normalization contract

Status: accepted
Date: 2026-08-16

## Context

Once several blocks exist over one `protid` index, someone will want a single
geometry from them. Combining representations is a solved problem with a
standard taxonomy, and the failure modes are equally well known.

The dominant failure is **scale**. A TM-score profile block has D ≈ N columns —
3,412 in a realistic run. A biophysics block has five: GRAVY, net charge, pI,
aromaticity, molecular weight. Concatenate them and run one PCA, and the
biophysics block contributes essentially nothing while the config file implies
the two are peers. The user gets a structure map labelled as a multimodal map.

The second failure is **silent zero contribution**: a block that is present in
the config, computed, weighted, and contributes ~0% of the final distance, with
nothing in the output saying so.

## Decision

**Four strategies behind one `FusionStrategy` protocol**, plus the no-fusion
default:

| Strategy | Joins at | Implementation | Default? |
|---|---|---|---|
| `none` | n/a | single block | — |
| `early` | feature space | standardize per block, concatenate, one PCA | no — **warns**, citing scale |
| `late` | distance space | `D² = Σ wᵢ · d̃ᵢ²`, `d̃ᵢ = dᵢ / mean(dᵢ)` | **yes, for fusion** |
| `graph` | affinity space | SNF (`snfpy`) or WNN per-protein weights | opt-in |
| `coregistered` | never | N independent spaces + cross-space diagnostics | **yes, overall** |

**The normalization contract: block scale is normalized before weighting, always,
with no way to opt out.** Each block's distances are divided by their mean, so
`mean(d̃ᵢ) = 1` for every block before any weight applies. Weights then mean what
a reader assumes they mean.

**Contribution share is a first-class, computed, recorded output.** For `late`,
the contribution of block *i* is

```
share_i = w_i · mean(d̃_i)² / Σ_j w_j · mean(d̃_j)²
```

It is computed, written to the manifest, logged at runtime, and **rendered on the
panel** in the explorer. Shares must sum to 1 under every strategy; this is
asserted, not assumed.

**Above ~70% share for any single block, warn loudly.** The map is that block's
map, and the output should say so rather than let the config's block list imply
otherwise.

**`early` warns with real numbers**, not a generic caution:

> Block `tmscore` (D=3,412) contributes 96.8% of total variance; block `biophys`
> (D=5) contributes 0.4%. Consider `strategy: late`.

**The weight vector is a displayed object**, not a buried config value. Every
fused output records it, and no fused map renders without it visible.

**Live weight-slider re-embedding is not offered.** It requires recomputing a
distance matrix and a projection, which is not feasible client-side at realistic
N. The explorer switches between **precomputed named presets** and says so. A
slider that silently snaps to the nearest preset is worse than labeled buttons.

## Consequences

- `late` is the recommended fusion path and the one that behaves sanely across
  wildly different D. Its cost is that it discards feature-space structure —
  you cannot read off which *dimension* drove a distance, only which block.
- `early` remains available because it is the standard baseline and reviewers
  will look for it, but it is never a default and never silent.
- `graph` (SNF) gives per-protein rather than global weights, which is its real
  advantage over `late` — a protein whose structure is well determined and whose
  function prediction is garbage gets weighted accordingly. Cost is one optional
  dependency (`snfpy`) and less interpretable weights.
- Every fusion path must produce contribution shares, so adding a strategy means
  implementing that computation. This is deliberate.
- Normalization by mean distance is undefined for a degenerate block where all
  distances are zero (one protein, or all-identical proteins). That case raises
  with a clear message rather than dividing by zero.

## Alternatives rejected

**No normalization; let users set weights to compensate.** Rejected: it makes
the weight vector meaningless across configs, since the same weights mean
different things depending on each block's D and scale. It also makes the
`biophys`-swamped-by-`tmscore` failure the user's problem to diagnose.

**Normalize by max distance instead of mean.** Rejected: maxima are
outlier-driven, so a single distant protein rescales the whole block.

**Z-score the distances.** Rejected: it produces negative distances, which is
meaningless, and breaks the metric properties the reducers assume.

**Only offer `late`.** Tempting, and it is the default for good reasons. Rejected
because `early` is the standard baseline a reviewer will expect to see
considered, and `graph` genuinely does something the other two cannot
(per-protein weighting). Offering them with honest diagnostics is better than
omitting them.

**Make fusion the default product.** Rejected — see ADR 0001. Fusing first
destroys the cross-space disagreement signal, which is the most interesting
output this work produces.
