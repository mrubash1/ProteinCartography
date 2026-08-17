# ADR 0008 — Cohort selection is a choice, not a given

Status: accepted
Date: 2026-08-16

## Context

Which proteins reach the map is decided upstream of every space, by a truncation
nobody sees:

```python
# download_pdbs.py:53-54
if maximum is not None:
    accessions = accessions[:maximum]
```

`max_structures` defaults to **5000** (`config.yml:62`). **No warning, log line,
or record is emitted when the truncation fires** — verified by reading the whole
path; the only prints are a tqdm label and a per-accession fetch error.

Before PR #106, the surviving set was chosen by Python's per-process hash
randomization: `aggregate_hits` accumulated into a `set` and wrote it in
iteration order. #106 fixed that by sorting — `aggregate_hits.py:55-56`:

```python
with open(output_file, "w+") as f:
    f.writelines(id + "\n" for id in sorted(id_set))
```

**But Phase 0 found that the sort does not survive.** The file `download_pdbs`
actually truncates is not that one. The chain is `aggregate_hits` →
`fetch_uniprot_metadata` → `filter_aggregated_hits` → `download_pdbs`, and
`filter_aggregated_hits` rewrites the list in the row order of
`uniprot_features.tsv`:

```python
# filter_aggregated_hits.py:92-93
with open(output_file, "w+") as f:
    f.writelines([protid + "\n" for protid in filtered_df["protid"]])
```

and that TSV is written in UniProt batch-response order
(`fetch_uniprot_metadata.py:255`), unsorted. So the in-code comment at
`aggregate_hits.py:51-54` — *"The ids are sorted because this file is truncated
to `max_structures` by `download_pdbs`"* — states an intent the code does not
achieve. **Truncation order currently depends on UniProt response ordering,
which this repo does not control.**

Even when the sort does apply, sorting is the wrong criterion. **Alphabetical
accession order is not random with respect to taxonomy**, because UniProt
accessions cluster by proteome and submission batch. Every clade-enrichment
result and evolutionary-rate estimate is conditioned on a possibly
taxon-biased sample, and nothing says so.

## Decision

**1. Fix determinism at the truncation point, not upstream of it.** The sort is
applied where the cut happens, so it cannot be lost by an intermediate rewrite.
This also fixes the current non-determinism, which #106 intended to fix and did
not.

**2. Selection becomes an explicit, configurable rule.**

```yaml
cohort:
  max_structures: 5000
  selection: significance      # "accession" (current, alphabetical) | "significance"
  significance_rule:
    blast: best_evalue_across_queries
    foldseek: best_tmscore
  record_truncation: true
```

**`selection: accession` is the default**, reproducing current behavior exactly,
because the default config must stay byte-identical. `significance` is opt-in.

**3. Truncation is recorded in the manifest and surfaced as a first-class
diagnostic**, never silent:

- candidate count before filtering
- candidate count after filtering
- candidate count after truncation
- whether truncation fired
- the rule and its parameters
- **taxonomic composition of retained versus discarded**

That last item is the one that matters scientifically. If truncation fired and
the discarded set is taxonomically different from the retained set, every clade
claim downstream is conditioned on that difference, and the user should see it
before believing a result.

**4. "Reproducible" and "principled" are tracked as separate properties.**
Alphabetical selection is reproducible and not principled. The manifest records
which rule ran so the distinction is auditable after the fact.

## Consequences

- The default map is unchanged, and the parity test proves it.
- Users who opt into `significance` get a cohort chosen by evidence strength
  rather than by accession string, at the cost of a different map — which is
  correct, and is why it is opt-in rather than a silent improvement.
- The truncation diagnostic will sometimes reveal that a published map was
  built on a taxon-biased sample. That is a feature.
- Cohort selection sits upstream of every block, so this is the one decision in
  the design that no downstream diagnostic can compensate for. It deserves its
  prominence.
- The determinism fix is a behavior change in the sense that output becomes
  stable where it previously depended on UniProt ordering. It cannot change a
  *correct* previous result into an incorrect one, but two runs that previously
  differed will now agree — which may surprise someone comparing to an archived
  run.

## Alternatives rejected

**Leave truncation as it is and document the caveat.** Rejected: it is currently
non-deterministic in a way that even PR #106's author believed was fixed. A
caveat in a README does not make a run reproducible.

**Always use significance ranking.** Rejected: it changes the default map, which
violates the byte-identical requirement and would make the parity test
impossible. It is the better rule and it is still opt-in, for this release.

**Remove `max_structures` and always take everything.** Rejected: the O(N²)
matrix and O(N³) PCA make an unbounded cohort a denial-of-service on the user's
own machine (ADR 0004). The bound is necessary; what was wrong was that it was
silent and arbitrary.

**Sample randomly with a recorded seed instead of ranking.** Genuinely
considered, and it has the virtue of being unbiased with respect to taxonomy
where alphabetical is not. Rejected as the default because a random subsample of
hits discards the strongest evidence as readily as the weakest, which is worse
for the primary use case. It is a reasonable third option to add later, and the
`selection` key leaves room for it.
