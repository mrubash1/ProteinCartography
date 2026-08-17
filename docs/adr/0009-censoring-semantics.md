# ADR 0009 — Censoring semantics: a zero is a missing measurement

Status: accepted
Date: 2026-08-16
Supersedes nothing. Written because Phase 0.5 claim A was **confirmed**.

## Context

`foldseek_clustering.py` builds the similarity matrix by filling every pair it
has no score for with a literal string:

```python
# foldseek_clustering.py:241-245
protid, targets_to_scores = protid_and_targets
scores = []
for target in targets:
    scores.append(targets_to_scores.get(target, "0.0"))
return [protid] + scores
```

The docstring states the intent plainly (`:237`):
*"This allows setting 0.0 as fillna(0.0) did with pandas."*

**Measured on a production run of 2530 proteins** (independently derived twice;
see `docs/REVIEW_LOG.md` Gate A):

| | |
|---|---|
| total cells | 6,400,900 |
| cells written `0.0` | 3,871,045 (**60.48%**) |
| cells written `0.000E+00` (a measured zero) | **0** |
| non-zeros per row | min 983, median 1000, max 1000 — **97.8% exactly 1000** |
| non-zeros per column | min 4, median ~896, max 2328 — no cap |

**The cause is `foldseek search --max-seqs`, default 1000**, which the pipeline
never overrides (`foldseek_clustering.py:87` passes only `-a`). The arithmetic
closes to the cell: predicted zeros `2530 × (2530 − 1000) = 3,870,900`, observed
`3,871,045`, and the 145-cell difference is exactly the shortfall of the 55 rows
that fell under the cap. The E-value default is **10**, not 0.001, so E-value is
not the binding constraint.

**Precise wording matters here.** A zero does **not** mean "Foldseek never
examined this pair" — with the default prefilter it examines every pair, by
k-mer matching and ungapped diagonal scoring. It means **"this pair lost the
per-query top-1000 cut"**. That is a *ranking* outcome on a competitive
per-query list, which is exactly why the same pair can be reported for query A
and dropped for query B, producing a 53.6% one-way rate.

**How wrong the fill is.** 925,435 censored cells have a measured mirror, so
their true value is knowable from the same file: median **0.772**, 96.4% above
0.5, 73.2% above 0.7. Meanwhile the lowest score Foldseek reports anywhere in
this matrix is **0.0549** and the median reported score is 0.801.

> **The reported-score distribution has no low end. Writing `0.0` does not merely
> lose information — it inserts a value Foldseek never produces.**

The distinction survives perfectly *in the file* — fills are the 3-character
`0.0`, measured values are 9-character `%.3E`. It is destroyed one step later by
the consumer, at `dim_reduction.py:57`:
`pivoted_df = pd.read_csv(pivot_file, sep="\t", index_col="protid")`, where
pandas coerces both to float64 zero.

**This is already producing wrong answers in shipped outputs**, not only in
hypothetical new ones:

1. `calculate_key_protid_tmscores.py` reuses the same `pivot_foldseek_results`
   with the same fill, and `calculate_concordance.py:61` computes
   `distance = tmscore - fident` with a default filter of strict `> 0`. A
   censored pair therefore yields `concordance = 0 − fident`, an artificial
   strong-negative reading as "sequence-similar but structurally divergent".
2. `plot_cluster_similarity.py:64-66,78-80` takes **means over the raw matrix**,
   so `*_leiden_similarity.tsv` and `*_strucluster_similarity.tsv` have
   between-cluster means dominated by fill. Between-cluster pairs carry the
   weakest scores and fall off the top-1000 list first, so at high censoring the
   surviving edges are disproportionately *within* cluster — clusters stay crisp
   while their arrangement decays into noise.

## Decision

**A zero in the TM matrix is treated as missing, not as measured dissimilarity.**

**1. Carry an explicit boolean mask.** `BlockResult.mask` is a `bool` array,
stored as `mask.npy` beside the values (ADR 0004). It is never a sentinel value
inside the float array — sentinels are how this problem happened.

**2. Build the mask by string form during parse, in `matrix_io.py`.** A cell is
censored iff its token is exactly `"0.0"`. This works on **any matrix this
pipeline has ever written, including archived output** — no Foldseek re-run and
no `.m8` files needed.

> This supersedes `PLAN.md`'s group 4 design, which proposed reconstructing the
> mask from the raw `.m8` alignment files. That is unnecessary for output this
> code path produced, and it is also the *wrong source* — the `.m8` files are
> search-mode web-API hits, a different artifact from the local all-versus-all
> run. The `.m8`/explicit-mask route survives only as a fallback for matrices
> this pipeline did not write.

**3. Expose censoring rate per protein as a diagnostic, and mark it
`fusable: false`** (ADR 0003) — it is a property of the measurement and it
correlates with length.

**4. Ship the censoring diagnostics suite**: per-protein rate, matrix zeroed
fraction, asymmetric-pair fraction, and **cross-cluster edge retention** — the
last being the non-obvious one that explains why heavily censored maps look
*more* structured, not less.

**5. Do not change the fill in the legacy path.** The default config must remain
byte-identical (parity test, commit group 5). The mask is carried *alongside*;
it never substitutes into the existing outputs. Mask-aware behavior is opt-in
via config.

**6. Symmetry is not assumed.** Even among both-reported cell pairs only 40.2%
are exactly equal (median difference 0.0014, max 0.6711), because TM-score is
length-normalized per query. Any `direct` representation records its
symmetrization choice in the manifest (ADR 0001).

## Consequences

- Downstream code can distinguish "not measured" from "measured as dissimilar"
  for the first time, and PCA can be run mask-aware where that matters.
- The mask costs one bool per cell — 6.4 MB at N=2530, and it compresses
  extremely well since it is ~60% one value.
- Archived runs become re-analyzable without recomputation, which is a larger
  practical win than it first appears.
- The censoring rate becomes a reportable property of every map, which will
  sometimes be unflattering. That is the point.
- Group 4's cost drops substantially versus the `PLAN.md` estimate, because
  string-form detection replaces `.m8` reconstruction.

## Alternatives rejected

**Treat zeros as measured dissimilarity (i.e. change nothing).** Refuted by
measurement: zero of 3,871,045 zeros is a measured value, and the 925,435 with
known mirrors have a median true score of 0.772.

**Impute the missing values.** Rejected for this PR. Imputation would put
modelled numbers into a scientific artifact, and the honest representation of
"we did not measure this" is a mask, not a guess. Imputation could be offered
later as an explicitly named, opt-in strategy — but it must never be the default
and must never be invisible.

**Raise `--max-seqs` so the problem goes away.** Rejected as out of scope and
not actually a fix: it changes existing behavior and output (violating the parity
requirement), it costs O(N²) alignment time, and at N=50,000 no setting makes the
matrix dense. Worth proposing upstream separately; it does not remove the need
for a mask.

**Use NaN instead of a separate mask.** Rejected: NaN propagates through
arithmetic in ways that silently poison reducers, and several downstream
consumers would need NaN handling added. A separate boolean array keeps the
value array numerically clean and makes mask-awareness an explicit opt-in at each
call site.
