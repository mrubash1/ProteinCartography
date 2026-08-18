# ADR 0011 — What co-registration compares, and what it refuses to average

Status: accepted
Date: 2026-08-17
Extends ADR 0001, which decided that co-registration is the default product.
Written when that decision had to be implemented and four sub-decisions turned
out to be load-bearing.

## Context

ADR 0001 settled the shape: several spaces over one protein index, and fusion
only when explicitly invoked. Building it raised four questions that the
abstraction does not answer, and each has a wrong answer that produces numbers
which look fine.

The pipeline already ships one cross-space comparison, `calculate_concordance.py`,
and it is instructive about how this goes wrong. It subtracts a fraction
sequence identity from a TM-score. Both are numbers in `[0, 1]`; neither is on
the other's scale. Their difference has no unit, no meaningful zero, and no
defensible comparison between two proteins, yet it is a column in the final
results and it reads like a measurement.

## Decision

### 1. The shared index is the intersection, and the loss is enumerated

The four blocks draw protids from three different files: `tmscore` from the
similarity matrix, `threedi` from the 3Di descriptor table, `biophys` and
`domains` from the UniProt features table. Different rules write those at
different points in a run, so their protein sets can differ.

Two spaces over slightly different sets still reduce, still plot, and still
look co-registered. Every per-protein comparison between them is then
conditioned on an overlap nobody chose, and nothing downstream can detect it.

We intersect rather than refuse, because a provider legitimately has no data
for some protein — no UniProt record, no foldable structure — and that is a
fact about the cohort. But the loss is **enumerated, not counted**: every
dropped protid is named in `coregistration/index.json` and on stderr. A count
tells you something happened; the list tells you whether it matters.

The one hard error is an empty intersection. That is not a small overlap, it is
the absence of anything to compare.

### 2. Neighborhood Jaccard, not a difference of scores

The replacement for `calculate_concordance.py` is a Jaccard index over
k-nearest-neighbor sets. It asks the question that was meant — do these two
kinds of evidence put the same proteins next to this one — and it asks it in a
quantity that is comparable across proteins, across spaces, and across runs,
because it is a ratio of set sizes rather than a difference of incommensurable
scores.

Rank correlation of the full distance profile is reported beside it, because
the two disagree usefully: a space can order distant proteins identically while
shuffling the near ones, which is the case that matters biologically and the
one Jaccard is sensitive to.

**`calculate_concordance.py` stays, and keeps its column.** Removing a
maintainer's existing feature inside a large PR is an unforced fight, and
invariant I6 promises the existing vocabulary keeps working. It gains a
docstring saying what the new metric is and why.

### 3. Ties are reported, never silently broken

When the k-th and (k+1)-th neighbors are equidistant, which one enters the set
is decided by protein index order. That is reproducible, which it must be, and
arbitrary, which no amount of care can fix.

So the count of boundary ties travels with every score. This is not defensive
decoration. On the eleven-protein demo the `families` space has **two distinct
points** — ten actins carry one Pfam family and the fusion carries two — so all
eleven rows tie, and its Jaccard against `structure` is largely a measurement
of protein index order. The score alone reads as a weak biological signal. The
tie count is what says it is not a biological signal at all.

The same demo has four proteins with byte-identical 375-residue sequences under
four accessions, so every sequence-derived space is degenerate on them too.

### 4. The comparison describes the geometry that is drawn, not the one declared

`spec.normalization` is recorded on every block and applied nowhere.
`spec.metric` is recorded and never consulted — `reduce_space` feeds features
straight into a euclidean PCA (FOLLOWUPS #29). Both are real defects.

Co-registration could honor either. It does not, and that is the decision:
applying a normalization here that `reduce_space` does not apply would make
this module describe a geometry **no map is drawn from**. A reader comparing
the disagreement metric against the plots would then be comparing two different
things, and nothing on either would say so.

So distances are euclidean over unnormalized features — exactly what the
reducer consumes — and both facts are stated in `geometry_caveats` on every
comparison and in the manifest. When the reducer becomes metric- and
normalization-aware, this changes with it, in the same commit.

### 5. Procrustes compares the pictures, and allows reflections

Procrustes needs both spaces in the same number of dimensions. Feature matrices
are not: `biophys` has four columns, `domains` one per observed family,
`tmscore` one per protein. The only shared dimensionality is the 2-D embedding.

So Procrustes is the one metric here computed on the plotted output rather than
on the geometry, it is the weakest of the three for that reason, and it is the
only one that answers "could a reader have superimposed these two plots by
eye". It is reported as such.

Reflections are permitted. UMAP and t-SNE output has no canonical handedness —
the same data with a different seed routinely returns a mirrored layout — so
forbidding reflection would report two identical maps as maximally different.

## Consequences

- A comparison cannot silently be about an overlap. It can still *be* about an
  overlap; it just cannot be about one silently.
- Every score is qualified by a diagnostic that can invalidate it. Consumers
  must read `boundary_ties` and `rank_correlation_undefined`, not just the mean.
- The metrics are numpy, with no scipy or scikit-learn dependency, per ADR 0006.
  Both Spearman and Procrustes agree with scipy's implementations to 3e-16,
  including Spearman on a 60%-censored matrix, where tie handling is the common
  case rather than an edge case.
- The euclidean-and-unnormalized decision is a **debt with a named creditor**.
  It is correct only for as long as `reduce_space` shares it.

## Alternatives rejected

**Refuse to co-register spaces whose protein sets differ.** Turns an ordinary
fact about the data into a hard failure, and the pipeline has legitimate
reasons to lose a protein between stages.

**Take the union and fill the gaps.** Filling means inventing coordinates for a
protein that was never measured in that space. `ProteinIndex.align` exists
specifically to refuse this; doing it here would defeat the module it depends
on.

**Report only the aggregate scores.** The per-protein tables are the entire
point of a disagreement mode: the interesting output is *which* proteins two
kinds of evidence disagree about, not how much they disagree on average.

**Suppress a score when ties dominate it.** Considered, and rejected as
paternalistic: the suppression rule would itself be an unexplained threshold.
Report the score, report the ties, let the reader decide.
