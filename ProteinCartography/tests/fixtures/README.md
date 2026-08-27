# Test fixtures

## The censoring fixture is SYNTHETIC, and that was a decision

`test_the_cap_alone_produces_the_crisp_cluster_trap` in
`../test_censoring_diagnostics.py` builds its own matrix: overlapping score
distributions, then each row keeps its top *k* partners and loses the rest —
which is exactly what foldseek's `--max-seqs` does. Nothing is placed by hand,
so the preferential loss of between-cluster edges falls out of the rule rather
than being assumed by the fixture.

It needs no file, runs in 0.01 s, and behaves identically in CI and locally.

## `fixture_tmscore_1400.accessions.txt` — the real matrix, and where it lives

A real 1400 × 1400 TM-score matrix over these accessions was built and verified
on 2026-08-26. **It is not in this repository.** It is 3.5 MB gzipped, Git LFS
cannot accept it (see below), and no test currently needs it. It lives in the
private notes repo at `evidence/pc041/`.

The accession list ships here because it is 16 KB and it is what makes the
matrix rebuildable: a seeded shuffle (seed 20260826) of the `chymo_full` cohort
in `docs/cohorts/`. Fetch those structures and run `rule foldseek_clustering`.

### What that real matrix has that the synthetic one does not

The cap signature on real data:

    1400 x 1400        zero fraction 28.6%
    measured per row   min 995 / median 1000 / max 1000
    rows at EXACTLY 1000   1386 of 1400  (99.0%)

plus real TM-score distributions and real asymmetry. The synthetic fixture
reproduces the *mechanism* and not the *data*, so anything about score values,
asymmetry, or the exact shape of the cap needs the real one.

### Why 1,400, and why not by shrinking a bigger matrix

The cap only engages above N = 1000, so a fixture that exercises it must be
larger than that. The obvious way to get one is to subsample a production
matrix down to N ≈ 1200–1500. **Measured, that does not work:**

| | zero fraction | measured per row (median) | cap at 1000 |
|---|---|---|---|
| production, N=2656 | 62.4% | **1000** | 2598 of 2656 rows exactly 1000 |
| subsample to N=1500 | 62.3% | 565 | **destroyed** |

A subsample keeps the zero *fraction* and loses the *cap*: a row capped at 1,000
of 2,656 columns, restricted to 1,500, keeps about 565. So the real matrix was
built by **running the search** over 1,400 structures.

### And once the cap binds, the censoring rate is not a free parameter

    censoring = 1 - 1000/N

so N=1,400 gives 28.6% and only N≈2,656 gives production's 62.4%. **The cap
signature and the censoring rate are different properties** and no fixture in
the 1200–1500 range carries both. §9 conflated them.

## Why not Git LFS

Tried, on 2026-08-26. GitHub refuses: *"can not upload new objects to public
fork"*. Forks have no LFS storage of their own — they bill against the parent —
so pushing needs write access to `Arcadia-Science/ProteinCartography`, which has
no `.gitattributes` and does not use LFS at all. Enabling it is an upstream
decision that every contributor would then inherit.

Tracked as **PC-042**, blocked.
