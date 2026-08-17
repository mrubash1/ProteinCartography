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

**1. Surface the problem by default; do not silently change the answer.**

An earlier draft of this ADR proposed sorting at the truncation point by default,
described as "reproducing current behavior". That was wrong, and checking the
code is what caught it. `fetch_uniprot_metadata` chunks the sorted accession list
into batches (default 100) and appends each batch's TSV response in the order
UniProt returns it, dropping accessions it does not recognize. So the order
reaching `download_pdbs` is *batch order with arbitrary order inside each batch,
and batch boundaries that shift whenever an accession is dropped*. Sorting at the
truncation point would therefore select a **different set of proteins** than the
pipeline selects today — a different map, and a failed parity test.

Silently improving a scientific result is still silently changing it. So the
default records the problem and changes nothing.

**2. Selection becomes an explicit, configurable rule, with the current
behavior named honestly.**

```yaml
cohort:
  max_structures: 5000
  selection: as_filtered   # "as_filtered" (current) | "accession" | "significance"
  significance_rule:
    measure: evalue        # "evalue" (default) | "bits" | "tmscore"
  record_truncation: true
```

**Correction, made during implementation: `foldseek: best_tmscore` is not
achievable, and the reason is circular.** The draft above asked for the best
e-value across queries for BLAST hits and the best TM-score for Foldseek hits.
The TM-score half cannot be done.

Search mode queries the Foldseek **web API**, and its `.m8` output has 21
columns with no alignment TM-score — `constants.FOLDSEEK_COLUMN_NAMES`, verified
against a recorded API response. The TM-scores this pipeline is built around
come from the *local* all-versus-all run in `foldseek_clustering`, which runs on
the downloaded structures. So ranking the cohort by TM-score would require the
structures that the ranking exists to choose. There is no ordering of the DAG
that resolves this; it is a property of where the measurement comes from.

What the web API does report is an e-value and a bit score, and those are what
`hit_significance.py` aggregates: the best value across every query that found
the hit, so a protein is ranked by the strongest evidence anyone has for it
rather than by the query that barely found it. `tmscore` remains in the
vocabulary because the measure is well-defined wherever scores exist — cluster
mode has real TM-scores — but it is not the default and cannot be used in search
mode.

One consequence worth stating plainly: **`significance` is a weaker rule than
this ADR originally implied.** An e-value is a sequence-and-structure alignment
significance, not a measure of structural similarity, so ranking by it selects
for confidently-detected hits rather than for structurally close ones. It is
still reproducible and still principled, which is more than either alternative
manages. It is not the TM-score ranking the draft promised.

| rule | order truncated | reproducible? | principled? |
|---|---|---|---|
| **`as_filtered`** (default) | UniProt response order, as today | **no** — depends on UniProt | no |
| `accession` | sorted by accession | yes | no — accessions cluster by proteome |
| `significance` | best e-value across queries | yes | yes, with the caveat above |

`as_filtered` is the default because it is what runs today, and the parity test
depends on that. It is named `as_filtered` rather than `accession` precisely
because calling it "accession" would repeat the mistake #106 made: assuming the
sort survived when it does not.

**3. The diagnostic fires regardless of the rule.** Whatever the selection, the
run records candidate counts, whether truncation fired, and — when the rule is
`as_filtered` and truncation fired — an explicit warning that the retained set is
not reproducible. A user does not have to opt into being told their cohort was
arbitrarily cut.

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
- Users who opt into `accession` or `significance` get a different, better-defined
  cohort, at the cost of a different map — which is correct, and is why both are
  opt-in rather than silent improvements.
- **The default remains non-reproducible, and this ADR does not fix that.** It
  makes it visible. Choosing `accession` is a one-line config change and is the
  right default for anyone starting fresh; making it the default here would
  break the parity test that the rest of this work depends on. Recommending the
  switch to upstream, separately, is the right venue.
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
