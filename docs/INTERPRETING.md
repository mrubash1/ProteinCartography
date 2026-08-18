# Interpreting a multi-space map

This document exists because the capability this PR adds is easy to over-read.
A map with four panels invites four times as many claims as a map with one, and
nothing about drawing a point at a coordinate makes the coordinate mean
anything. Every section below is of the form *this is licensed, that is not,
and here is the number that tells you which*.

The short version: **the diagnostics are not decoration and they are not
optional.** They run for every space, unconditionally, because a caveat you have
to opt into is read by exactly the people who already suspected the problem
(ADR 0014). Several of them exist specifically to tell you *not* to read
something. When one fires, it is the finding.

---

## 1. What a distance in one of these maps is

Every space reduces to two dimensions for display, and **the 2-D coordinates are
a picture of the geometry, not the geometry**. The metrics that qualify a space
are computed in the full-dimensional space, before reduction.

**Licensed.** *Cluster compactness.* Whether a group of proteins is tight
relative to its distance from other groups is computed in the matrix, and it
survives projection well enough to be read off a picture.

**Not licensed.** *Ratios of distances on the plot.* "A is twice as far from B
as from C" is a statement about the projection, not about the proteins. UMAP in
particular does not preserve distance ratios and does not claim to; it preserves
neighbourhoods, approximately, and `trustworthiness`/`continuity` tell you how
approximately (§3).

**Not licensed, and the most common error.** *"These two are close, so they
share a function."* In a TM-score geometry this is specifically wrong, not
merely unsupported: protease clans sit above TM 0.7 across entirely different
substrate specificities. Structural similarity is evidence about fold, and fold
is only sometimes evidence about function.

**Recorded on every comparison** (`coregistration.GEOMETRY_CAVEATS`): distances
are euclidean because `spec.metric` is recorded on a block and consulted by
nothing that reduces it, and features are unnormalized because
`spec.normalization` is likewise recorded and applied nowhere. Those are honest
descriptions of the geometry the map is drawn from rather than the one the
config declares. They are `docs/FOLLOWUPS.md` #29 and #32.

**And with `representation: direct`, the distance may not be a distance.** That
mode reads the similarity matrix as a matrix and hands `1 - TM` downstream. Its
config gate checks that rows and columns line up, which is necessary and is not
sufficient: alignment makes the matrix mean its labels and says nothing about
whether those numbers can be laid out in a Euclidean space. Largely they cannot.
Double-centring the squared distances of the published 160-protein actin cohort
puts **44.6%** of the positive spectral mass into negative eigenvalues, so a
large share of what the matrix asserts is not embeddable in any number of
dimensions — and every reducer consuming it discards that silently.

`diagnostics/metricity.py` measures this and records it on the block, under
`derived.metricity`. **Read it as context, not as a verdict.** It deliberately
sets no threshold: measured on sub-cohorts of that same single family the
statistic runs from 15.3% to 49.3%, so any warning level fitted to it today
would be a constant derived from one protein family (`docs/FOLLOWUPS.md` #49).
The report carries its own convention — whether squared or raw distances were
centred, and whether the negative mass is divided by the positive mass or the
total — because those two choices move the same cohort between 19.9% and 44.6%,
and a bare percentage is not comparable to another bare percentage. It carries
the cohort size and censoring rate for the same reason.

---

## 2. A zero in the similarity matrix is missing, not dissimilar

This is the single most consequential thing to know about ProteinCartography's
input, and it is measured rather than argued.

On a production 2530-protein matrix, **60.5% of cells are the literal string
`"0.0"` and none of them is a measured zero.** `foldseek search` runs with
`--max-seqs` defaulting to 1000, so each query reports at most its top 1000
partners and the rest are fill. 97.8% of rows have exactly 1000 non-zero
entries. Rows are uniform at the cap and columns are not, which is the signature
of a per-query top-N cut rather than of biology.

**What follows:**

- A zero means *Foldseek did not report this pair within the per-query top-1000*.
  Reading it as "these proteins are structurally dissimilar" is a category error.
- The matrix is **not symmetric**: 36.6% of reported pairs are reported in one
  direction only, with a maximum asymmetry of 0.9944.
- **Censoring is not uniform across the map.** Between-cluster pairs carry the
  weakest scores and fall off the cap first, so at high censoring nearly every
  surviving edge is *within* a cluster. The clusters look crisper while their
  arrangement decays into noise. This is the number worth reading in the
  censoring section: `cross_cluster_edge_retention`.

The mask is carried explicitly through `matrix_io` and recoverable from any
existing matrix by string form, because fills are `"0.0"` and measured scores
are `%.3E` (ADR 0009).

---

## 3. The nine diagnostics, and what each one licenses

Every space writes `spaces/{space_id}/diagnostics.json`. A section is absent
when this run could not answer it, and the set that landed is recorded in the
manifest, so absence is information rather than silence.

### Censoring — was the input measured?

Per-protein rate, matrix zeroed fraction, asymmetric-pair fraction, and
cross-cluster edge retention. Reported per block, and only for a *profile*
block, where the pairwise questions are well posed.

*Reads:* a high censoring rate with low cross-cluster retention means the
between-group structure is the least reliable thing on the map — exactly the
part a reader is most likely to interpret.

### Cohort — was the selection fair?

Candidates before and after truncation, the selection rule, whether truncation
was reproducible, and the taxonomic composition of retained against discarded
proteins.

*Reads:* if truncation fired and the discarded set is taxonomically skewed, every
"this clade is absent" claim is about the cohort rule, not about biology. Note
the default rule is `as_filtered` rather than a sort, because #106's sort does
not survive UniProt (ADR 0008) — so **truncation order is not reproducible
across runs** unless you set one.

Cluster mode writes no cohort report. That is correct rather than missing: it
makes no cohort decision, because the user supplies the structures.

### Redundancy — do the blocks say different things?

Pearson and Spearman correlation of the *pairwise distance* upper triangles
between every pair of blocks in a space. Scale-invariant, so it can be read
before a fusion strategy is chosen.

*Reads:* two blocks correlating above 0.90 are close to the same measurement
wearing two names, and fusing them 50/50 double-counts one signal. This is the
number a contribution share cannot give you: a 50/50 share is correct *and
uninformative* for two copies of the same block.

### Contribution share — what did each block actually put in?

Both the requested weight vector and the *realized* share, which are different
numbers. Shares sum to exactly 1 under all four strategies.

*Reads:* under `early`, a block's share is essentially its column count, because
standardizing equalizes columns rather than blocks. If one block is 96% of the
geometry, the map is that block's map and the others are perturbing it. Prefer
`late`, which normalizes the blocks (ADR 0002, ADR 0013).

### Trustworthiness and continuity — did the map survive two dimensions?

Per protein, and **never averaged together**, because they fail in opposite
directions. Trustworthiness penalizes neighbours the 2-D layout *invented*;
continuity penalizes true neighbours it *tore apart*.

*Reads:* T ≫ C means the layout is over-compressed — it has crushed distinct
regions together. C ≫ T means it is torn — it has separated things that belong
together. A protein flagged in either direction should not have its position on
the plot read at all.

### Neighborhood stability — is this neighbour list a finding or a coin flip?

Per-protein Jaccard overlap between the k nearest in the data and the k nearest
under a perturbation: the cohort resampled, and Gaussian noise added at 10% of
the median pairwise distance.

*Reads:* **check `informative` first.** When `k` is at least half the candidates
a replicate offers, a "neighbourhood" is most of the cohort and the score cannot
discriminate — it returns 1.0 by construction. On the eleven-protein demo it is
`false` for all seven spaces. Where it *is* informative, a protein at or below
0.30 has a neighbour list that is close to a coin flip and should not be read as
a finding.

### Cluster stability tree — is the partition real, or is it the resolution knob?

Leiden across a range of resolutions, with the adjusted Rand index between
adjacent rungs.

*Reads:* a **plateau** — a range of resolutions all recovering the same grouping
— is a level in the data. Uniformly low adjacent agreement means the number of
clusters is tracking the parameter and no particular partition of that space
should be reported as *the* partition. If every resolution returns the same
count, the structure is either very strong or the sweep is too narrow to have
moved anything, and the report says which it cannot distinguish.

### Negative controls — what does nothing look like?

Two, failing in different ways on purpose. `shuffled_labels` permutes the
assignment over the same distances, holding cluster count and every cluster size
fixed. `random_distances` fits the same Leiden to a random matrix of the same
shape.

*Reads:* **the second one is the uncomfortable number and it is the point.** A
clustering fitted to pure noise returns a genuinely positive silhouette. On the
demo that baseline is 0.105: `structure` clears it at 0.503, and `fused_early`
does not meaningfully at 0.176. A partition that does not beat its own noise
control is not a finding, however good the picture looks.

A control that was requested and could not run is listed under `skipped` with
the reason, rather than omitted — a shorter list would read as "ran and found
nothing".

### Self-diff determinism — would a second run agree?

Two runs at the same versions on an N>750 fixture, asserted identical, in CI as
a standing guard.

*Reads:* this is a guard rather than a per-run number, but the reason it exists
is worth knowing. Before PR #106, `sklearn`'s PCA switched to a randomized
solver above 500 rows and drew from the global numpy random state, so **every
map above ~500 proteins was irreproducible.** A fixture at or below 500 proteins
cannot exercise that at all.

---

## 4. Comparing two spaces

`coregistration/summary.tsv` carries four cross-space metrics per pair.

- **Neighborhood Jaccard at k**, per protein. The principled replacement for
  `calculate_concordance.py`, which subtracts fraction sequence identity from
  TM-score across non-commensurate scales. The old column is kept for
  compatibility and marked superseded; it is not a quantity anyone should read.
- **Rank correlation** (Spearman) of full distance profiles, per protein.
- **Procrustes disparity** between the 2-D embeddings — absent rather than
  fabricated when the two spaces were not reduced by the same reducer.
- **Cluster-assignment ARI** between the two spaces' own partitions.

**Read the ARI column carefully, and note when it is blank.** It is withheld
whenever either space has fewer than two clusters, because the adjusted Rand
index of two single-cluster partitions is **1.0 by convention** and of one
degenerate side against a real partition is **0.0** — both are conventions
rather than measurements. On the demo, seven of ten pairs are blank for exactly
this reason, and before that guard existed the table read "families and
fused_late agree perfectly" about two spaces that had each found nothing.

**The circularity warning.** `struclusters` is a structurally-gated graph
clustered on *amino-acid identity* — `foldseek clust` receives the alignment
database and cannot accept the TM-score database. So a "structure versus
sequence" contrast built on `struclusters` as the structural reference is partly
circular. It is on the not-fusable list for that reason (ADR 0003), and any
comparison using it as a reference should say so.

---

## 5. Cluster-level enrichment

`enrichment/` holds an FDR-corrected tidy table rather than only an SVG.

**Every row describes the `structure` space's partition and says so**, because
that was the pipeline's only clustering when the table was built (ADR 0012 §1).
A row is not a claim about a `physicochemistry` cluster.

Two behaviours to know:

- A group with no data for a column is reported **`untested`, with the reason**,
  never compared against a substituted zero. The legacy
  `plot_cluster_distributions.py` takes the other branch: its `remove_nans`
  substitutes a synthetic `0` for an all-missing cluster and then runs a
  Mann-Whitney against it, so a cluster whose structures all failed to download
  reads as *significantly lower pLDDT* rather than as *no measurement*
  (FOLLOWUPS #34).
- EC number and subcellular localization are named in the plan and have **no
  data source**: `fetch_uniprot_metadata` requests neither. Taxon and domain
  architecture are present and are what the demo enriches on (FOLLOWUPS #35).

---

## 6. Reproducibility, honestly

- **Default configuration output is byte-identical** to upstream at the commit
  this branch forked from, checked by a parity test that CI runs on every pull
  request and that is itself mutation tested (12 mutations detected, 5
  survived-as-expected with a recorded reason, 0 unexplained holes). Read that
  mutation result narrowly: the harness runs pipeline steps and diffs output
  trees, and all 17 of its mutations name six pipeline scripts, so it says
  nothing about the block, fusion or diagnostics modules.
- **Two runs of the same config agree above N=500**, guarded in CI.
- **The Leiden partition is not reproducible across environments at very small
  N.** Two environments agreeing on scanpy, leidenalg, igraph, numpy and
  scikit-learn, and differing only in scipy 1.13.1 against 1.15.2, return
  different two-cluster memberships at N=11 and identical ones at N=250. At
  eleven proteins the kNN graph is nearly complete and Leiden's optimum is
  degenerate, so `arpack` decides the tie. This applies to the pre-existing
  `leiden_clustering` rule too; `envs/analysis.yml` does not pin scipy
  (FOLLOWUPS #42).
- **The parity test cannot see what the default configuration does not run.**
  Significance-ranked cohort selection, for example, is covered by unit tests
  instead. And because both sides of the comparison run in one environment, it
  cannot see an environment-dependent output either.

---

## 7. If you read one thing

Look at the diagnostics before the picture. In the shipped eleven-protein demo,
every one of the seven spaces reports `informative: false` for stability,
trustworthiness runs 0.34 to 0.76, four to ten of the eleven proteins are flagged
as having unreadable positions, and seven of ten space pairs cannot report an
ARI at all.

That is the correct answer for eleven proteins, and it is the reason the demo
ships with those numbers visible rather than hidden. **A map that looks like a
finding and a map that is one are the same picture.** The difference is in this
file's numbers.
