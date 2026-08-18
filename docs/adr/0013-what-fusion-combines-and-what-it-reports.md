# ADR 0013 — What fusion combines, and what it reports

Status: accepted
Date: 2026-08-17
Extends ADR 0002, which chose the taxonomy. Written when that taxonomy had to be
implemented, and five of its details turned out to be either underspecified or
wrong.

## Context

ADR 0002 settled which strategies exist — `none`, `early`, `late`, `graph` —
and the contract they owe: block scale normalized before weighting, contribution
shares computed and summing to 1, a loud warning above 70%, and a weight vector
that is displayed rather than buried.

`reduce_space.py` refused a multi-block space outright until now, so none of
that had ever run. Implementing it raised five questions ADR 0002 does not
answer, and answered two of ADR 0002's own claims differently than ADR 0002
does. Both kinds are recorded here; §3 and §5 are corrections rather than
extensions, and ADR 0002 has been amended to point at them.

## Decision

### 1. `late` and `graph` emit a profile, not a metric

`early` produces a feature matrix and reduces like any other space. `late`
produces an `(N, N)` distance matrix and `graph` an `(N, N)` affinity matrix,
and neither is a feature matrix. The obvious move is a metric-aware reducer —
`sklearn`'s `metric="precomputed"` — and we do not make it.

Instead each protein's **row** of the fused matrix is its feature vector. That is
exactly the representation the pipeline has always used for the TM-score matrix
(`representation: profile`, ADR 0007), reached from a different direction, so it
is neither novel nor a workaround. What it buys is that every space in the
system still goes through one reducer core. Two reducer paths — one for
feature-space strategies and one for distance-space strategies — would
eventually disagree, and a disagreement between two PCAs looks exactly like a
scientific result.

`FusionResult.representation` records which of the three a given matrix is,
because it is not recoverable from the array.

### 2. A multi-block space is co-registered before it is fused

Fusion combines row `i` of each block as one protein and has no way to check
that claim. The blocks are built by different rules from different files, so
their protein sets and their orders can differ — on the eleven-protein demo
cohort, `tmscore` and `biophys` genuinely list the same proteins in different
orders.

So `reduce_space` takes the intersection, in the first listed block's order,
aligns every block to it through `ProteinIndex.align`, and names every dropped
protein on stderr. This is ADR 0011 §1 applied *inside* a space rather than
between spaces, and for the same reason: a fused geometry conditioned on an
overlap nobody chose looks exactly like one that is not.

### 3. Two contribution shares, because ADR 0002's cannot say anything

ADR 0002 defines `late`'s share as

```
share_i = w_i · mean(d̃_i)² / Σ_j w_j · mean(d̃_j)²
```

and the normalization contract makes every `mean(d̃)` exactly 1 by construction.
The formula therefore returns the normalized weight vector. It cannot report
anything the config did not already state, which is not what "contribution share
is a first-class, computed, recorded output" was meant to mean.

What a block actually contributes to a fused squared distance is
`w_i · mean(d̃_i²)`, and `mean(d̃²) = 1 + var(d̃)` — so a block whose distances
are widely dispersed puts more into the geometry than its weight suggests, and
one whose pairs are all equally far apart contributes scale and no shape.

Both are reported. `share` is ADR 0002's formula, kept because it is the
contract and because computing it from the *measured* means rather than from the
weights makes it a live check on the normalization: it stops being the weight
vector the moment normalization breaks. `realized_share` is the fraction of the
fused quantity the block accounts for, and it is what the dominance warning
keys on.

They are not a formality. On the demo cohort `late` over `tmscore` and `biophys`
at equal weights reports a nominal 50/50 and a realized 34/66.

### 4. `graph` is SNF in numpy, and its two departures are named

Wang et al. 2014 (Nature Methods 11:333). The scaled exponential kernel (eq. 3),
the full normalization (eq. 8), the local k-nearest normalization (eq. 9) and the
cross-diffusion update are the paper's. Two things are not:

- **The cross-view average is weighted** by the configured block weights, which
  the paper has no notion of.
- **The full normalization is re-applied after every update** rather than only
  at the start, keeping each iterate row-stochastic so that the per-step delta
  is a comparable quantity across steps. That is what makes the non-convergence
  warning meaningful rather than decorative.

`snfpy` is not installed and ADR 0006 keeps it that way, so there was no
reference output to compare against. The tests check the algorithm's properties
instead — row-stochastic iterates, symmetry, convergence, two identical blocks
splitting every protein at exactly 0.5, an informative block beating a noise
block for every protein, and the geometry moving monotonically with the weights
in both planted partitions of the fixture at once.

One further departure is forced by ADR 0002 rather than chosen. **The published
kernel is not scale-invariant**: its exponent is `d²/(μ·ε)` with `ε` in units of
distance, so multiplying a block by 1000 changes its affinities entirely.
Normalizing to unit mean distance before the kernel — which the contract
requires regardless — removes that, and `graph` reproduces a rescaled block's
geometry to 1e-17.

### 5. Per-protein weights live on the output, not in the config

ADR 0002 names per-protein weighting as `graph`'s unique advantage over `late`.
`SpaceSpec.weights` is one scalar per block, so there was nowhere to put them,
and FOLLOWUPS #16 recorded the ADR as overclaiming.

The resolution is that the ADR was describing the wrong end. Per-protein weights
are not something a user configures — nobody knows, per protein, which evidence
to trust. They are something SNF *produces*: the fused network gives every
protein its own mixture of the blocks, recovered afterwards as
`BlockContribution.per_protein_share`. With an informative block against a pure
noise block, both weighted 1.0, `late` reports 50/50 by construction while
`graph` earns 59.7/40.3, and does so for each protein individually.

`early` and `late` report `None` rather than an array of identical numbers,
because an array of identical numbers is a scalar pretending to be a
measurement.

### 6. A parameter the strategy will not read is a config error

`SpaceConfig.params` was free-form. `STRATEGY_PARAMS` now declares what each
strategy consumes and the validator rejects anything else at parse time.

This is the third rule of the same kind (`spec.metric` #29, `spec.normalization`
#32) and the first one enforced structurally rather than by remembering. It
found an instance immediately: `test_config_schema.py`'s example config had
carried `params: {K: 20}` on a `graph` space since the schema was written —
uppercase, matching the paper, read by nothing.

## Consequences

- Fusion is reachable from a config and is still never a default. ADR 0001's
  ordering is unchanged: co-registration is the product, fusion is an analysis
  you ask for.
- Every fused output carries its weight vector and both shares in
  `manifest_*.json` under `extra.fusion`, and prints them to stderr while it
  runs. ADR 0002 required the weight vector be "a displayed object"; this is the
  display for a run with no browser.
- `coregister.py` rebuilds each compared space's geometry rather than reading it
  back, because what is written to disk is the 2-D embedding and two of the
  three metrics need the full-dimensional matrix. For a fused space that means
  running the fusion twice per run. Deterministic, so it is the same answer, but
  `graph` re-runs its diffusion — worth revisiting if fusion becomes the
  expensive step.
- The dominance warning keys on the realized share while the config sets the
  nominal weight, so a config can look balanced and still trip it. That is the
  intended behaviour and it is why both numbers are printed side by side.
- A pairwise block still cannot enter a fused space. It carries condensed
  distances rather than features, and `late` would be the natural home for it;
  the existing "use representation: profile" error is unchanged. Recorded as a
  followup rather than built, because no provider currently produces one.

## Alternatives rejected

**A metric-aware reducer, so `late` could hand its distances to UMAP directly.**
Rejected: it is a second reducer path through the system, and the property that
stops two PCAs drifting apart is that there is only one. The profile
representation is already the pipeline's idiom for exactly this shape, and it
costs an `(N, N)` intermediate that `late` had already built.

**`snfpy` as an optional dependency.** Tempting, because it would give a
reference implementation to test against. Rejected under ADR 0006: `graph` would
then be the one strategy that silently disappears in the bare environment, and a
fusion strategy that exists on some machines is worse than one that exists
everywhere and is 60 lines of numpy. The cost is real and is stated in §4 — the
tests check properties rather than agreement.

**Report only the realized share.** It is the honest number and the one the
warning uses. Rejected because ADR 0002's formula is the published contract, and
silently computing something else is the same failure as a manifest field
nobody honors, inverted: the value the document promises would not be the value
the code produces.

**Let `params` stay free-form and validate inside each strategy.** Rejected:
the strategy runs hours after the config is parsed, and by then the misspelling
has already been recorded in a manifest that looks configured.

**Fuse without intersecting, and require that blocks match.** Rejected for the
reason ADR 0011 gives: a provider legitimately drops a protein it has no data
for, and that is a fact about the cohort rather than a failure. What must never
happen is losing those proteins without saying so.

**Weight blocks per protein from the config.** The literal reading of ADR 0002's
claim, and it is unimplementable in any useful sense — it asks the user to know,
per protein, which evidence deserves trust. That knowledge is the output of the
analysis, not an input to it. See §5.
