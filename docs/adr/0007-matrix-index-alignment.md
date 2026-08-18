# ADR 0007 — Every labeled-matrix load asserts index alignment

Status: accepted
Date: 2026-08-16

## Context

`foldseek_clustering.py` writes the all-versus-all similarity matrix by
iterating a target collection to produce the header, and separately iterating
the entries to produce the rows:

```python
# foldseek_clustering.py:261-268
with open(output_file, "w", newline="") as fh:
    csv_writer = csv.writer(fh, delimiter="\t")
    header = ["protid"] + [f"{column_prefix}{target}" for target in targets]
    csv_writer.writerow(header)
    for entry in sorted(entries.items()):
        csv_writer.writerow(get_line_for_protid(entry, targets))
```

Before PR #106, `targets` was a `set`. Rows were `sorted()`; columns were not.
Python salts string hashing per process and nothing sets `PYTHONHASHSEED`, so
the column order was a per-run random permutation of the row order.

This is not hypothetical. Measured on a production run of 2530 proteins:

```
row labels == col labels (as sets)   : True
row order  == col order (exact)      : False
columns sitting in their row position: 2/2530 (0.08%)
positional diagonal == 1.0           : 2/2530 (0.08%)
label-aligned diagonal == 1.0        : 2530/2530 (100.00%)
```

**Reading that matrix positionally returns the wrong cell 99.92% of the time.
Reading it by label is exact.** Two independent runs of the same input differ
only in header order: identical row-label sequences, different column order,
and zero differing cells once canonically sorted.

PR #106 fixes the ordering by sorting `targets`. That fix is necessary and it
is in this branch. It is not sufficient as a durable guarantee, for three
reasons:

1. It is one line in one function. Nothing prevents the next writer of a
   matrix-shaped artifact from reintroducing the defect.
2. Existing on-disk outputs produced before #106 are still permuted. Any
   analysis pointed at archived output inherits the defect silently.
3. The failure is silent by construction. A permuted matrix has the correct
   marginal distribution, the correct value set, and the correct labels. Only
   the pairing is wrong, and nothing downstream currently checks the pairing.

The existing pipeline survives this defect by accident rather than design: PCA
over a consistently permuted set of columns is invariant to the permutation, so
the shipped UMAP is not garbage. That accident does not extend to any
representation that treats the matrix as a matrix — notably `1 − TM` as a
metric distance.

## Decision

**Every load of a labeled matrix goes through one helper,
`ProteinCartography/matrix_io.py`, which reads the header and the row labels and
asserts they are the same sequence in the same order.**

The helper:

- raises with a readable diff naming the first divergent position, the row
  label found there, and the column label found there;
- offers opt-in repair by reordering columns to row order, emitting a loud
  warning that names PR #106 as the cause;
- refuses to return a bare `numpy` array — the caller receives labels alongside
  values, so positional indexing requires deliberately discarding them.

**The assertion stays permanently, after #106 and after any future fix.** It is
cheap (one comparison of two label lists) and it is the only thing that catches
the next occurrence.

**`representation: direct` (`1 − TM` as a metric distance) is gated behind a
passing assertion** and a config flag. On unfixed output it is silently,
catastrophically wrong, and the 0.08% figure above is the measure of how wrong.

**Scope: new consumers only.** The five existing raw-matrix consumers
(`dim_reduction`, `leiden_clustering`, `get_source_of_hits`, and
`plot_cluster_similarity` in two rules) were each audited and are all already
label-safe — they use `index_col="protid"`, whole-frame operations, or a
merge-transpose-merge that recovers column labels. Retrofitting them would touch
five existing files to buy assertion coverage rather than a bug fix. That trade
fails the minimize-footprint invariant, so it is recorded in `docs/FOLLOWUPS.md`
instead.

## Consequences

- One new module, no existing file changed. Purely additive.
- Any future block provider that reads a matrix inherits the check for free.
- A permuted-column test fixture is added to the suite; it must fail the loader.
  This is a positive control on the check itself — an assertion nobody has seen
  fire is an assertion nobody knows works.
- Loading cost rises by one list comparison per load. Irrelevant next to parsing
  an O(N²) TSV.
- The audit above is a point-in-time result. If a future change makes an
  existing consumer positionally-indexed, this ADR's scope decision should be
  revisited.

## Alternatives rejected

**Trust PR #106 and add nothing.** Rejected: #106 fixes the producer, and the
defect class is silent-wrong-answer. The producer fix does not protect archived
output, does not protect against a second producer, and leaves no positive
signal that alignment held on any given run.

**Assert only inside the block providers.** Rejected: the assertion belongs at
the I/O boundary, so that it fires once per load rather than once per consumer,
and so that adding a consumer cannot forget it.

**Normalize on write instead of checking on read.** Rejected as insufficient
alone — it is what #106 already does. Checking on read is what makes the
guarantee verifiable at the point of use, and it covers inputs this project
does not produce.

**Drop the labels and standardize on positional order everywhere.** Rejected:
this is the failure mode, not the fix. Labels are the only thing that made the
production matrix recoverable.
