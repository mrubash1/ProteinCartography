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

### Why the measure is an e-value and not a TM-score

The draft of this ADR specified `significance_rule: {foldseek: best_tmscore}`.
It ships ranking by e-value instead. This section went through two versions and
the first one was wrong, so both are recorded.

**First correction, and it overclaimed.** It said a TM-score is unobtainable
because the pipeline's TM-scores come from the *local* all-versus-all run in
`foldseek_clustering`, which operates on the downloaded structures — so ranking
the cohort by TM-score would need the structures the ranking exists to choose.
That circularity is real, but it applies to the **all-versus-all matrix**, and
cohort ranking does not need the matrix. It needs one score per candidate
against the query proteins, which is a different and much cheaper thing.

**What is actually true.** `foldseek_apiquery.py` accepts `--mode tmalign`, and
in that mode the web API *does* return a TM-score per hit — query versus
candidate, before anything is downloaded. Verified by a live query with the demo
actin structure against `afdb-swissprot`: 938 hits, top hit actin at TM 0.9999,
bottom hit an unrelated GTP pyrophosphatase at 0.402. So a TM-score-ranked
cohort is feasible. It is not what ships, for three reasons, and the third is
the serious one.

1. **The Snakefile never passes `--mode`,** so the pipeline always runs `3diaa`.
   Making the mode configurable is a change to search behavior in its own right.
2. **Switching modes changes which hits come back and in what order,** so it
   changes the cohort, and therefore the map. That cannot land under the
   byte-identical requirement no matter how much better the ranking is.
3. **The two modes are indistinguishable from their output.** The server returns
   the *same 21 columns in the same positions* and puts the TM-score where the
   e-value goes; `bits` becomes roughly TM×100. Nothing renames, nothing is
   flagged, and the mode is not recorded anywhere in the results. Any code
   reading the column named `evalue` silently gets a quantity with the opposite
   polarity.

That third point is not hypothetical. On the live response above,
`extract_foldseek_hits.py`'s default filter — `evalue < 0.01` — keeps **0 of 938
hits**. And a significance ranking that trusted the column name would order the
cohort worst-first, selecting the pyrophosphatase over the actin.

`hit_significance.py` therefore **refuses** tmalign-shaped input rather than
guessing at it: bounded in [0, 1], never small, and bit scores at TM-score scale
together mean the file cannot be interpreted with confidence. Refusing is the
right call while the mode is unrecorded. **The real fix is to record the mode**
next to the results, at which point interpreting either mode correctly becomes
trivial and TM-score ranking becomes available as an opt-in.

`tmscore` stays in `SIGNIFICANCE_MEASURES` because the measure is well-defined
wherever scores exist — cluster mode has real TM-scores — but it is not the
default and nothing in search mode can currently supply it safely.

One consequence worth stating plainly: **`significance` as shipped is a weaker
rule than this ADR originally implied.** An e-value is alignment significance,
not structural similarity, so it selects for confidently-*detected* hits rather
than structurally close ones. It is still reproducible and still principled,
which is more than either alternative manages. It is not the TM-score ranking
the draft promised, and the path to that ranking is now written down.

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
