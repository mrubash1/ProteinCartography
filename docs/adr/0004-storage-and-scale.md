# ADR 0004 — Block storage format and the O(N²) ceiling

Status: accepted
Date: 2026-08-16

## Context

The pipeline's central artifact is a dense N×N TSV,
`all_by_all_tmscore_pivoted.tsv`. Measured on a production run of 2530
proteins it is 40.8 MB.

Two facts about that file determine the storage design.

**It is mostly not data.** Foldseek's local `search` runs with `--max-seqs`
defaulting to 1000, so each query reports at most 1000 partners and everything
else is written as the literal string `0.0` (see ADR 0009). At N=2530, 60.5% of
cells are fill. The censoring fraction is `1 − min(1000, N)/N`, so:

| N | censored | dense TSV | float64 in RAM |
|---|---|---|---|
| 500 | 0% | 2.5 MB | 2 MB |
| 2,530 | 60.5% (measured) | 40.8 MB | 51 MB |
| 5,000 (shipped default `max_structures`) | ~80% | ~130 MB | 200 MB |
| 50,000 | ~98% | ~10.3 GB | 20 GB |

**The file grows as O(N²) while the information in it grows as O(1000·N) —
linear.** At N=50,000, 99.2% of the file is a single repeated fill token. Only
the N=2530 row above is measured; the rest follow from the confirmed cap
mechanism, which reproduces the measured row to two decimal places.

Two further costs compound this. PCA is forced to `svd_solver="full"`
(`dim_reduction.py:76`) for determinism, which is O(N³) and requires the dense
matrix resident. And `pandas.read_csv` peaks at roughly 2–3× the final frame
size while parsing.

The practical ceiling today is therefore around N≈5,000 — which is exactly the
shipped default for `max_structures`. The pipeline is running at its limit by
default.

## Decision

**Blocks and spaces are stored as `numpy` binary, not TSV.**

```
OUTPUT_DIR/blocks/{block_id}/
├── features.npy | distances.npy    # float32; condensed upper triangle for pairwise
├── mask.npy                        # bool, censoring (ADR 0009)
├── protids.txt                     # canonical order, one per line
└── manifest.json
```

Specifically:

- **`float32`, not `float64`.** TM-scores are reported by foldseek to four
  significant figures (`7.512E-01`). `float32` carries ~7 decimal digits, so it
  is lossless with respect to the source precision and halves memory.
- **Condensed storage for symmetric pairwise blocks** (upper triangle only),
  matching `scipy.spatial.distance.squareform`. Halves it again.
- **The censoring mask is a separate `bool` array**, not a sentinel value inside
  the float array. Sentinels are how the current `0.0` problem happened.
- **`protids.txt` is the canonical order** and is the only thing that defines
  row/column identity. Consistent with ADR 0007: identity lives in labels, never
  in position.

Combined, a pairwise block is ~8× smaller in memory than the dense float64 TSV
equivalent, before any sparsity is exploited.

**Existing TSV outputs are untouched.** `all_by_all_tmscore_pivoted.tsv` and
every `final_results/` artifact keep their current format and filenames. The
`.npy` store is strictly additive, in new directories.

**We do not implement a sparse or out-of-core representation in this PR.** The
censoring mask makes the matrix's sparsity explicit and machine-readable, which
is the precondition for a sparse backend, but building one now would be
speculative: no user has asked for N > 5,000, and the reducers, the fusion
strategies, and `plot_cluster_similarity` would all need sparse-aware rewrites
to benefit. The block store's interface returns arrays through
`ProteinCartography/spaces/store.py`, so a sparse backend can be added later
behind it without changing any provider.

**We state the ceiling rather than hiding it.** Every new component documents
its behavior at N=500 / 5,000 / 50,000, and the store raises a clear error with
the measured memory estimate rather than dying in an allocator.

## Consequences

- Memory headroom improves roughly 8× for pairwise blocks; the working ceiling
  moves from N≈5,000 to somewhere near N≈15,000 without further work.
- `.npy` is not human-inspectable. Mitigated by `manifest.json` beside every
  array carrying shape, dtype, checksum, and provenance, and by keeping the
  legacy TSV outputs unchanged for anyone who wants to read numbers by eye.
- Cache invalidation is by input hash recorded in the manifest, not by mtime,
  so a rerun with identical inputs is free and a rerun with changed inputs is
  correct.
- N=50,000 remains out of reach. That is honest and documented, not solved.

## Alternatives rejected

**Keep everything as TSV.** Rejected on measurement: 10.3 GB at N=50,000, and
99.2% of it a repeated fill token. Also loses dtype and the mask.

**Parquet for everything.** Parquet is a good fit for the tabular per-protein
artifacts (`neighbors_k{K}.parquet`, `stability.parquet`) and is used for those.
It is a poor fit for a dense numeric matrix: columnar encoding of an N×N float
block adds metadata overhead per column, and at N=50,000 that is 50,000 column
chunks. `.npy` is the right shape for arrays, Parquet for tables. We use both,
each where it fits.

**HDF5 / Zarr.** Rejected for this PR as a dependency cost with no present
payoff. Either becomes attractive at the point an out-of-core backend is
actually needed, and the `store.py` interface leaves that door open.

**Sparse (CSR) as the primary representation now.** Rejected as premature. It is
the obvious next step and the mask is designed to enable it, but it would
require sparse-aware reducers and fusion to deliver any benefit, which is a
large speculative change inside an already-large PR.
