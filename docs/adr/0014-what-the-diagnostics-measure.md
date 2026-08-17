# ADR 0014 — What the diagnostics measure, and what they refuse to

**Status:** accepted, 2026-08-17 (commit group 8b)
**Supersedes:** nothing. **Amends:** nothing.
**Related:** ADR 0002 (fusion and normalization), ADR 0004 (storage and scale),
ADR 0009 (censoring semantics), ADR 0013 (what fusion reports).

---

## Context

Group 8a made every fused map carry its weight vector and contribution shares,
which is what ADR 0002 promised. Running the demo with them made something
plain: a map can be fully provenanced and still be unreadable, and nothing in
the provenance says so. The shares describe how the numbers were combined. They
say nothing about whether combining was worth doing, whether the input was
measured, or whether the projection to two dimensions kept anything.

Phase 5 lists nine diagnostics. This ADR covers the ones group 8b built —
block redundancy, embedding faithfulness, and the wiring of censoring and
cohort reporting — and records the decisions that were not obvious.

---

## Decision 1 — Diagnostics are not opt-in

The rule is gated on `spaces` being defined, not on a `diagnostics:` key.

A caveat somebody has to opt into is read by exactly the people who already
suspected the problem. The reader who most needs to be told that the map places
unrelated proteins next to each other is the reader who trusts the map, and
that reader will not enable a diagnostics flag.

**Cost, stated:** every space costs one extra rule and one extra pass over its
distance matrix, which is `O(N²)` in memory and `O(N² log N)` for the sorts.
On the largest matrix measured (2530 proteins) that is 51 MB and a few seconds.
At the ADR 0004 ceiling of N≈15,000 it is 1.8 GB, which is the same order as
the map itself and is the point at which this decision should be revisited.

**Rejected:** a `diagnostics.enabled` flag defaulting to true. It is the same
thing with an escape hatch, and the escape hatch would be used by exactly the
runs that produce inconvenient numbers.

---

## Decision 2 — Two statistics for the projection, not one

Trustworthiness and continuity are both reported, always, and never averaged
into a single "quality" number.

They measure different failures. Trustworthiness asks about neighbors the map
*invented* — proteins adjacent on the page that are far apart in the data.
Continuity asks about neighbors the map *lost*. Only the first produces wrong
biology, because only the first asserts something; the second hides something
true, which is a smaller sin and has a different remedy.

Averaging them would destroy exactly the information a reader can act on. A map
at trustworthiness 0.98 and continuity 0.62 has torn one group into two and
should have its splits distrusted; a map at 0.62 and 0.98 has crowded distinct
groups together and should have its proximities distrusted. Their mean is 0.80
in both cases and tells you to do nothing in particular.

**Rejected:** reporting only trustworthiness, on the grounds that it is the one
that produces false claims. It is, and the demo shows why that is not enough:
`fused_late` scores 0.34 and 0.24, and the second number is what says the
splits in that picture are not real either.

---

## Decision 3 — Per protein, with the global value as their exact mean

Both statistics are computed per protein and the summary is their arithmetic
mean, which is *exactly* the standard global definition rather than an
approximation of it.

This is a property worth having rather than a coincidence. The normalizing
constant `k(2N-3k-1)/2` is the largest rank excess a single protein can
accumulate, so dividing by it puts every per-protein value in `[0, 1]` and
makes the mean of them equal the usual formula. Without that, "this protein's
position is unreliable" would be a share of a number about the whole map rather
than a statement about the protein, and the per-protein table would be
decoration.

**Consequence, and it is the reason the tests use a relative tolerance:**
summing per protein and averaging associates the arithmetic differently from
one global sum, so the result differs from scikit-learn by up to one unit in
the last place — measured 2.2e-16 relative, one machine epsilon, in 5 of 12
comparisons and exactly zero in the other 7.

---

## Decision 4 — Redundancy compares distances, and is scale-free by construction

Block redundancy correlates the two blocks' **pairwise distances**, not their
features, and reports both a Pearson and a Spearman coefficient.

Features cannot be compared: blocks have incommensurate dimensionality — 200
columns against 4 in the fixture, thousands against a handful in practice — and
what two blocks share is a protein index, not a column space. What each block
says about every *pair* of proteins is the common currency.

Both coefficients are invariant under multiplying either block's distances by a
positive constant. That is load-bearing rather than incidental: it means the
answer cannot depend on ADR 0002's unit-mean-distance step having run, which is
what lets the diagnostic report *before* a fusion strategy has been chosen —
which is what Phase 5 asks for. The fixture pins it: `narrow` and
`narrow_rescaled` differ by a factor of 1000 and correlate at exactly 1.

Reporting both is not redundancy of its own. Pearson asks whether the blocks
agree about how far apart things are, Spearman only whether they agree about
the ordering, and a pair high on the second and low on the first ranks the same
proteins as similar on different scales — which is the pair most worth fusing,
and the case the normalization contract exists for.

**Rejected:** the Székely distance correlation (dCor). It detects nonlinear
dependence, which sounds strictly better, and it answers a question nobody
here is asking — two blocks can be strongly dependent and still disagree about
every neighbor. The question is whether fusing changes the map, and that is a
question about agreement, not about dependence.

---

## Decision 5 — Redundancy is what a contribution share cannot say, and the two are reported together

A share of 50/50 is *correct* for two identical blocks: half the numbers did
come from each. It is also useless, because the same map would have appeared
with either block deleted. Nothing in ADR 0002's machinery can reveal that, so
redundancy sits in the same report and its warning says so in as many words.

The demo makes the case: `tmscore` and `biophys` correlate at Spearman 0.883,
just under the 0.90 threshold, while the three fused spaces report an honest
50/50. Both numbers are right and only the pair of them is informative.

---

## Decision 6 — `k` is clamped to the cohort, and the request is kept

A cohort smaller than the configured neighborhood size is ordinary rather than
an error: the demo has eleven proteins and the default `k` is fifteen. The
statistic is undefined for `k >= (2N-1)/3`, so `k` is reduced to the largest
value that works, the original request is recorded beside it, and the reduction
is reported as a warning.

This is the idiom `reduce_pca` already uses for `n_components` and
`ReducerResult.params_used` already established: record parameters *after*
clamping, because two runs with identical configs and different N are not the
same run and a manifest that reports only the request cannot tell them apart.

**Rejected:** refusing a `k` that does not fit. That would mean the smallest
cohorts — the ones whose maps are least trustworthy, and the demo among them —
are the only ones that get no diagnostics at all.

---

## Decision 7 — Censoring is reported only where it is defined

`diagnostics/censoring.py` was built in group 4 and had no caller. Group 8b
wires it in rather than rebuilding it, and wires it in **conditionally**.

A block's censoring channel is per cell of that block's features. The pairwise
reports — asymmetry, cross-cluster edge retention — are only meaningful when
those cells *are* protein pairs, which is true for a profile block, whose
features are the similarity matrix itself, and false for anything else. The
condition is checked rather than assumed, and a block that fails it gets a
recorded reason rather than a silently absent section.

Cross-cluster edge retention additionally needs a partition, which is why the
rule takes `--clusters`. Today that comes from the legacy Leiden path, which is
the pipeline's only clustering; when spaces cluster in their own right it
should take that instead.

---

## Decision 8 — Item 9 is a test, not a report

The self-diff determinism check produces no output and appears in no manifest.
What it guards against is not a property of anyone's data: two runs of the same
pipeline on the same input produce two maps that both look right, and nothing
in either one says the other exists. There is nothing to put in a report.

It carries its own negative control, because a determinism guard is the easiest
kind of check to write in a form that cannot fail. scikit-learn selects its SVD
solver from the input shape, and the randomized solver it picks above 500
samples is the only one that is nondeterministic without a seed — so the same
guard on a 400-protein fixture passes unconditionally. One test configures PCA
the way this repository deliberately does not and *requires* the two runs to
differ; if it ever starts passing trivially, everything around it has become
decoration and its failure message says so.

---

## Decision 9 — A `DiagnosticsConfig` field must be read or must say why not

Enforced by a test that enumerates the dataclass's fields and requires each to
be either mentioned by a module outside `config_schema.py` or listed in an
exemption table with the Phase 5 item that will read it. It fails in both
directions: an unlisted dead field fails, and a listed field that comes alive
also fails, so the table shrinks as the remaining diagnostics land instead of
going stale.

This is ADR 0013 §6's `STRATEGY_PARAMS` idea in the one form available here.
Fusion parameters can be validated, because which parameters a strategy reads
is a fact about the config vocabulary. Whether a diagnostic reads a field is a
fact about the code, so it is checked by looking at the code.

It earned itself immediately, in the same way `STRATEGY_PARAMS` did: writing it
surfaced that `from_legacy` silently dropped the whole `diagnostics:` key on the
legacy branch, three lines below a comment explaining why `enrichment` must not
be dropped there for exactly that reason. A cluster-mode config could set
`diagnostics.k` and be ignored, or misspell it and not be told.

---

## What this deliberately does not do

- **No single quality score.** See decision 2.
- **No threshold that claims authority.** `FAITHFUL_THRESHOLD` and
  `REDUNDANT_THRESHOLD` are reporting conventions and say so where they are
  defined. The literature offers no threshold for either statistic, and
  inventing one and then quoting it would be worse than naming where the line
  was drawn.
- **No imputation and no repair.** A diagnostic that fixed what it found would
  make the next run's diagnostics describe the fix.
- **Nothing that needs a per-space clustering.** Phase 5 items 7 and 8 are
  diagnostics *about* a partition, and spaces do not cluster yet. Deferred to
  group 8c with the decision they depend on.
