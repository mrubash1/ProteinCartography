# ADR 0012 — What cluster enrichment tests, and whose clusters it tests

Status: accepted
Date: 2026-08-17
Companion to ADR 0011, which decided what co-registration compares. This decides
what a *single* map's clusters are compared against, which turned out to raise a
different set of questions.

## Context

The pipeline already asks what a cluster is made of, and answers it only as a
picture. `plot_cluster_distributions.py` runs a Mann-Whitney per cluster per
numeric column and draws a star on a violin plot when the corrected p-value
clears a threshold. The numbers behind the stars are never written down, so
nothing can sort them, join them, or check them; and the categorical
annotations — lineage, protein family — are not tested at all, though they are
the columns a biologist actually asks about.

PLAN Phase 6 names four categories to enrich on: taxon, EC, domain
architecture, localization. Building it raised five questions the phase
description does not answer, and each has a wrong answer that produces a table
which looks right.

## Decision

### 1. Enrichment takes a cluster table by path, and names the clustering in every row

The honest version of this feature enriches *each space's* clusters. It cannot
be built yet, because spaces do not cluster: `reduce_space` emits coordinates
and nothing else, and the pipeline's only clustering is Leiden over the TM-score
matrix, in the legacy path.

The tempting move is to reach into that clustering and present the result as the
map's. We do not, because that is the shape of the defect this whole work
exists to correct — `struclusters` is amino-acid identity presented as a
structural label (FOLLOWUPS #9), and a TM-score clustering presented as *the*
clustering of a four-space run is the same error one level up.

So the module is indifferent to where clusters come from. `enrich_clusters`
takes `--clusters PATH` naming any table with a protid column and a cluster
column, and writes `clustering` into every output row. Today the Snakefile
points it at `leiden_features.tsv` and the table therefore describes the
`structure` space. When spaces gain their own clustering, that is a wiring
change and a second `clustering` value, not a rewrite.

Adding a clustering algorithm is deliberately **not** part of this. It is a
separate concern from a statistic, it would need a numpy-only reimplementation
of something the pipeline already gets from scanpy, and inventing a novel
clustering inside a large PR is an unforced expansion. It is also what PLAN
Phase 6's cross-space ARI needs and does not have, so the two arrive together
or not at all.

### 2. Two tests, because there are two kinds of annotation

Continuous columns get the two-sided Mann-Whitney U with the tie-corrected
normal approximation — the generalization of what the plotting script already
does. Two-sided, because a cluster of unusually short proteins is as much a
finding as a cluster of long ones.

Categorical columns get the **one-sided** hypergeometric tail, computed exactly.
One-sided is a decision rather than an oversight. An enrichment table answers
"what is this cluster made of", and the two-sided p-value on a term is dominated
by terms that are rare everywhere: it would fill the table with rows observing
that a cluster of 50 does not contain a family only 3 proteins carry. Depletion
is not lost — every row carries observed and expected counts and a fold
enrichment, which is below 1.0 exactly when the term is under-represented.

Both are numpy plus `math.lgamma` and `math.erfc`, per ADR 0006, and both are
cross-checked against scipy behind an `importorskip`: Mann-Whitney to 2e-16
including on heavily tied and 60%-censored input, the hypergeometric to a
*relative* 6e-12 down to p = 1e-94, Fisher's one-sided exact test to 8e-15, and
Benjamini–Hochberg to 2e-16.

### 3. The universe is the annotated background, and the shrinkage is reported

For a categorical column, the universe is the proteins whose cell is present,
not every protein in the cohort. A protein whose annotation was never determined
is not evidence that it lacks a term, and counting it as one biases every term
toward whichever cluster happens to be better annotated — annotation
completeness correlates with taxonomy, which is one of the things being tested.

This cuts harder than it looks, and the reason is the file format rather than
the statistic. A protein with no Pfam families is written as an empty field,
which `read_csv` returns as NaN, so **after a round-trip through TSV "carries no
families" and "was never annotated" are the same bytes.** On the 400-protein
test cohort that takes Pfam's universe from 400 to 232 and raises every fold
enrichment accordingly.

It is still the right choice — an enrichment background is conventionally the
annotated background — but it is not free, so it is made visible three ways:
`n_universe` is on every row, the per-column universe is in the manifest, and
any column that loses proteins says so on stderr.

### 4. The correction family is one annotation column

Benjamini–Hochberg is applied per annotation column, not across the whole table.
The columns have wildly different vocabulary sizes — a lineage has a dozen
terms, an InterPro column has thousands — and pooling lets the large vocabulary
set the correction for the small one, so a finding's q-value would depend on
which *other* columns happened to be configured. Every row carries the counts
and the family is one column, so a reader who wants a global correction can
recompute it; the reverse is not recoverable.

Terms carried by fewer than `min_term_count` proteins are not tested. A term
carried by one protein cannot reach significance however concentrated it is —
its smallest attainable p-value is the cluster's share of the universe — and
testing it anyway costs every other term in the family a larger correction. The
count of dropped terms is in the manifest, not nowhere.

### 5. A test that could not run is a row, with a reason

Every result carries a `note`. When it is set, the test did not run, the p-value
is NaN, and the row **does not enter the correction family** — counting an
untested hypothesis would deflate every q-value in the table, and nothing on the
blank row would say so.

This is the branch `plot_cluster_distributions.remove_nans` takes the other way.
Its `default=(0,)` fires whenever a cluster's values for a column are all
missing, so a cluster whose structures all failed to download is tested as a
distribution of one synthetic `0.0` against every other cluster's real values,
and reads as significantly low confidence rather than as unmeasured
(FOLLOWUPS #34). An untested row must never be indistinguishable from a null
one.

## Consequences

- An enrichment table is auditable row by row: `n_cluster`, `n_universe`,
  `n_term_cluster` and `n_term_universe` reconstruct the exact test, so a reader
  can check a p-value without rerunning anything.
- The table is sorted by q ascending with a full deterministic tiebreak, so it
  is readable at the top and byte-stable at the bottom. Two runs over the same
  input produce the same bytes.
- **Two of PLAN's four categories have no data source.** `fetch_uniprot_metadata`
  never requests `ec` or `cc_subcellular_location`, and adding them would change
  the columns of `uniprot_features.tsv` and therefore of
  `aggregated_features.tsv` — precisely what the byte-identical guarantee
  forbids (FOLLOWUPS #35). Enrichment is column-driven rather than
  category-driven, so both arrive the moment the columns do, and a requested
  column that is absent is named on stderr and in the manifest rather than
  skipped.
- With exactly two clusters every comparison appears twice, as mirror images
  with the same p and opposite effect. They are reported rather than collapsed,
  because a reader scanning by cluster wants a row for their cluster; the
  duplication does not change the q-values, because Benjamini–Hochberg gives
  tied p-values the same q.
- Nested vocabularies make the correction conservative. `Chordata` implies
  `Eukaryota`, so the two are not independent hypotheses and both will be
  enriched wherever one is. This is a property of the annotation, not of the
  test, and it is not corrected for.

## Alternatives rejected

**Cluster each space here, so enrichment describes the map.** The right end
state and the wrong commit. It means a numpy-only clustering algorithm, chosen
and justified, inside a PR whose reviewability is a hard requirement — and the
enrichment statistic would then be untestable independently of it. §1 makes the
switch a wiring change instead.

**Extend `plot_cluster_distributions.py` rather than add a module.** It would
have to keep drawing the plot, gain categorical tests, and gain a table, in a
file that imports matplotlib and `arcadia_pycolor`. The statistic would then be
unavailable in any environment without a plotting stack, and untestable without
one. The plotting script is left exactly as it is (invariant I6).

**Use scipy, which has all four of these.** ADR 0006. The framework has to
import and run with only numpy and pandas, and this is the module a maintainer
would most reasonably expect to need scipy — which is what makes it the one
worth not needing it.

**Report only the significant rows.** The rows that failed are the evidence that
the ones that passed are not everything. `significant` is a column, not a filter.

**Treat every protein in the cohort as the universe.** Simpler, and it makes
completeness of annotation into signal: a cluster that is better annotated
appears enriched for everything. See §3.
