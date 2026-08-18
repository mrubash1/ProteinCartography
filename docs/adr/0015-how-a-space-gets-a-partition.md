# ADR 0015 — How a space gets a partition, and what the three remaining diagnostics measure

Status: accepted
Date: 2026-08-18

## Context

Phase 5 lists nine diagnostics. Group 8b built the six that need no clustering.
The three left — neighborhood stability, the cluster stability tree, and the
negative controls — could not be built, because **no space had a partition**.
`reduce_space` emits coordinates and nothing else, and the pipeline's only
clustering is Leiden over the TM-score matrix in the legacy `leiden_clustering`
rule. That single partition is a partition of the `structure` space, and it was
being used to answer questions about all seven (FOLLOWUPS #41).

So this ADR answers one architectural question and three measurement questions.

## Decision 1 — a space is clustered by scanpy's Leiden, not by a reimplementation

Every other numeric component in this work is hand-written numpy: the three
co-registration metrics (ADR 0011), four enrichment statistics (ADR 0012), four
fusion strategies including SNF (ADR 0013), trustworthiness and continuity (ADR
0014). Leiden is not in that class.

**It adds no dependency.** `scanpy=1.9.3` and `leidenalg=0.9.1` are already
pinned in `envs/analysis.yml`, already run in the default DAG, and `rule
diagnose_space` already declares that environment. ADR 0006 governs *optional
and licensed* dependencies; this is neither. The decision costs the maintainer
nothing they are not already paying.

**A diagnostic about a partition must be about the partition that ships.** This
is the argument that actually settles it. A hand-rolled clusterer would make the
resolution sweep sweep an algorithm that never produces `leiden_features.tsv`,
and the negative controls controls for a method nobody runs. Being
*approximately* the pipeline's clustering is worse here than not clustering at
all, because the numbers would look comparable to the shipped ones and would not
be.

**Leiden is not SNF.** Group 8a hand-rolled similarity network fusion in sixty
lines of numpy because it has a closed form and because an optional `snfpy`
would have made a fusion strategy vanish in the bare environment (ADR 0013 §2).
Leiden has a local-moving phase, a refinement phase and an aggregation phase,
and its refinement guarantee — that every returned community is internally well
connected — is the entire reason it is preferred over Louvain. An approximation
carrying the name would be worse than no implementation.

**The cost, and what pays it.** Hand-writing bought a reference implementation
to check against, and group 8b's exact agreement with scikit-learn's
trustworthiness was the strongest single piece of evidence in that group. Here
the reference *is* the implementation, so there is nothing to compare arithmetic
against. What replaces it is a cross-**path** check rather than a
cross-implementation one: `clustering.leiden_partition` and
`leiden_clustering.scanpy_leiden_cluster` are required to return **identical
labels** for the same matrix, verified at N=250 and again above 500 where
scanpy's graph construction changes. `clustering._clamped` duplicates the legacy
clamping rather than importing it — importing would pull scanpy in at module
scope and break the bare environment — and that agreement test is the only thing
keeping the duplication honest.

**Availability degrades rather than fails.** The import is deferred,
`is_available()` reports in the shape ADR 0006 rule 2 specifies, and a space
that cannot be clustered loses the partition-dependent sections and keeps
everything else. Argument validation and the below-three-proteins short circuit
come *before* the availability check, because a two-protein space has one
cluster by definition and needing scanpy installed to be told so would make an
optional dependency load-bearing for an answer that does not depend on it.

## Decision 2 — stability perturbs the cohort and the scores, and the reference moves with the subsample

Phase 5 item 3 specifies three perturbations. Two are implemented and one is
not.

**Resample the retrieved cohort**, without replacement. The proteins in a run
are a retrieval result, not a fixed population. Without replacement rather than
a true bootstrap: duplicated proteins sit at distance zero from each other and
would occupy the whole neighborhood.

**Add score noise**, Gaussian and symmetric, scaled to a fraction of the
matrix's median pairwise distance so that it means the same thing on a TM-score
block and on a physicochemistry block — ADR 0002's incommensurability problem,
met in a different place.

**Jitter the top-k cutoff per query is not implemented.** It is defined only on
a censored profile block, and after fusion a space's features may be anything;
running it over an arbitrary feature matrix would produce a number that looks
right and means nothing. Recorded rather than approximated.

**The reference neighbor set is recomputed inside each subsample**, not taken
from the full cohort. This is the choice that makes the rest interpretable. Both
sides lose the same proteins and both promote the same replacement, so a
replicate with no noise scores **exactly 1.0** and every departure is
attributable to the noise term alone. It also keeps `subsample_fraction` live
rather than decorative: a smaller cohort has fewer candidates competing for the
k-th slot, so the same noise flips fewer of them.

**The default noise is 0.10 and it is a stress level, not an error estimate.**
Nothing here could know a TM-score's measurement error. The question the
diagnostic answers is "would this neighborhood survive a perturbation of this
size", and a perturbation too small to move anything makes every protein look
determinate.

**`k` is clamped to `round(f*N) - 1`, and a clamp is not enough.** Group 8b
shipped `DEFAULT_K = 15` against an 11-protein demo and broke all seven spaces
while every unit test passed at N=240. The ceiling here is tighter, because k
must fit the subsample rather than the cohort — and running the demo showed that
clamping *silently* is its own defect. At N=11 the clamp lands on 8 of 9, so
every protein's k nearest are all the others, and the Jaccard is 1.0 whatever
the noise. All seven spaces reported perfect stability under a sigma half the
size of the data. So the report carries `informative`, false when k is at least
half the candidates a replicate offers, with a warning naming the fraction. A
statistic with no room left to be wrong in reporting 1.000 is the most confident
possible way to say nothing.

## Decision 3 — the sweep reports a plateau, not a cluster count

Leiden returns more clusters as its resolution rises, always. A single run
therefore cannot distinguish "the data has this many groups" from "the parameter
was set here". Sweeping and measuring the adjusted Rand index between *adjacent*
resolutions can: a plateau, where a range of resolutions all recover the same
grouping, is a level in the data; uniformly low agreement means the cluster
count is tracking the knob.

`PLATEAU_THRESHOLD` is 0.80 and is a reporting band in the sense ADR 0014 uses —
a statement of where a line was drawn, not a threshold from any literature. A
sweep of one resolution is **refused** by the config validator, because it has
no adjacent pair and therefore measures nothing; so are duplicate resolutions,
for the same reason.

ARI and the silhouette are hand-written numpy and agree with scikit-learn to
1e-12 relative, including in the degenerate cases where its answer is a
convention rather than a formula: ARI of two all-in-one-cluster partitions is
1.0 by definition and 0/0 by the formula, and the pipeline reaches that case
whenever `leiden_clustering` short-circuits below three proteins.

## Decision 4 — two negative controls, and they fail differently on purpose

"If the pipeline produces attractive clusters on noise, and it will, everyone
should see that first."

**`shuffled_labels`** permutes the assignment over the same distances, holding
the cluster count and every cluster size fixed. It isolates correspondence from
every marginal that could produce separation by itself, and it needs no
clusterer, so it is computed inside `diagnostics/partition.py`.

**`random_distances`** fits the same Leiden to a random matrix of the same shape
and scores the result. This is the uncomfortable one, and it is worth the cost
of needing a clusterer: it returns a genuinely positive silhouette. On
`fusion_cohort`'s structureless block a fitted clustering scores *higher* than
the block's own correct partition does.

Both are named in `diagnostics.negative_controls` and validated against a set
enumerated in `diagnostics/partition.py`, imported by the config validator the
way `STRATEGY_PARAMS` is from `fusion` (ADR 0013 §6). A control named in a
config and implemented nowhere would be silently skipped, which is
indistinguishable from one that ran and found nothing.

**A control that cannot run is named, not omitted.** The demo produced exactly
this failure: one space's random matrix clustered into a single group, which has
no silhouette, so that space's report carried a shorter control list than its
six neighbours. A reader comparing them cannot tell "ran and found nothing" from
"never ran", and the shorter list reads as the former. Requested controls that
could not be produced now appear under `skipped` with their reason.

## Decision 5 — a space prefers its own partition, and says when it could not have one

`diagnose_space` now clusters the space and writes `clusters.tsv` beside the
report. Cross-cluster edge retention uses that partition rather than the legacy
structural one, which closes FOLLOWUPS #41.

When a space cannot be clustered the legacy partition is still used, and the
report records `partition.source` and a `caveat` saying that every
partition-dependent number below describes the structural clustering applied to
this space's distances rather than this space's own grouping. ADR 0012 §1 makes
the same move for the enrichment table, which describes the `structure` space
and says so in every row.

`clusters.tsv` is **not** a declared snakemake output. Whether it exists depends
on scanpy being importable and on the space having three proteins, and a rule
that promises a file it cannot always write fails the run instead of degrading
it.

## Consequences

- Six of Phase 5's nine diagnostics became nine. `NOT_YET_CONSUMED` is empty and
  FOLLOWUPS #36 is discharged: every `DiagnosticsConfig` field is read by
  something, proved by a test that parses for the attribute access.
- The default DAG is unchanged at 16 rules, the search DAG at 25, and the
  multispace demo at 35. No new rule; `diagnose_space` does more inside the one
  it already had.
- A reviewer who objects to scanpy in the diagnostics path can delete
  `clustering.py`, three sections of `diagnose_space.py` and two config fields.
  Nothing else imports it, which is the droppability ADR 0006 asks of every
  optional capability, applied to one that is not optional.
- **The partition is not reproducible across environments at very small N**, and
  neither is the pre-existing `leiden_clustering` rule's. Two environments
  agreeing on scanpy, leidenalg, igraph, numpy and scikit-learn, and differing
  only in scipy 1.13.1 against 1.15.2, give different two-cluster memberships at
  N=11 and identical ones at N=250. `envs/analysis.yml` does not pin scipy.
  Recorded as FOLLOWUPS #42; not caused by this work, only made visible by it.

## Alternatives rejected

**Hand-roll a numpy clustering.** Rejected on Decision 1's second argument: a
diagnostic about a partition that the pipeline does not produce is not a
diagnostic. The dependency argument, which looked decisive at first, turned out
not to matter either way — scanpy is already there.

**Make clustering optional and degrade to no partition at all.** This is what
happens when scanpy is absent, and it is the right *fallback*, but it would be
the wrong *design*: it would leave FOLLOWUPS #41 open, since the legacy
partition would remain the only one available in the environment the pipeline
actually runs in.

**Take the partition as an input rather than computing it.** `diagnose_space`
already accepts `--clusters`, and for the censoring section that is enough. It
cannot work for the resolution sweep, which must re-cluster at every rung.

**Report the stability value without qualifying it.** Rejected after running the
demo. See Decision 2.
