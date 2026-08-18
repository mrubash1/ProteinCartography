# The multi-space demo, walked through

Eleven actin structures — the same inputs as the cluster-mode demo — mapped
seven ways, compared pairwise, and diagnosed.

```bash
snakemake --configfile demo/multispace/config.yml --use-conda --cores 8
```

Offline, about two minutes after the conda environments exist, 35 rules.

**Everything the legacy pipeline produces is still produced, byte for byte.**
The `blocks:` and `spaces:` keys are additive. Delete them and this config is
the cluster-mode demo.

---

## What it builds

**Four blocks**, each one measurement of the same eleven proteins:

| block | what it measures | needs |
|---|---|---|
| `tmscore` | the existing all-versus-all TM-score matrix | foldseek (already a dependency) |
| `threedi` | Foldseek 3Di k-mer profiles — local backbone geometry | foldseek |
| `biophys` | isoelectric point, GRAVY, aromaticity, charge per residue | nothing |
| `domains` | InterPro/Pfam architecture | nothing |

**Seven spaces.** Four co-registered — `structure`, `local_structure`,
`physicochemistry`, `families` — one per block, so each is a different opinion
about the same proteins. Then the same two blocks (`tmscore` + `biophys`) fused
three ways: `fused_early`, `fused_late`, `fused_graph`. Three strategies over
one pair of blocks is deliberate: it is the only way to see that the strategy,
not the data, is what changes.

**Ten pairwise comparisons**, in `coregistration/summary.tsv`.

**Nine diagnostics per space**, in `spaces/{space_id}/diagnostics.json`.

---

## What it reports, and why that is the point

**This demo's headline result is a verdict against itself.** Eleven proteins
cannot support these maps, and the diagnostics say so in numbers rather than the
demo quietly omitting them. That is the behaviour worth demonstrating: the same
pipeline on a real cohort produces the same fields, and you read them the same
way.

Measured on this cohort:

- **Stability reports `informative: false` for all seven spaces.** `k` clamps
  from 15 to 8, and an 80% subsample holds 9 proteins, so every protein's eight
  nearest are *all* the others and the Jaccard is 1.0 whatever the noise. Every
  space scored a perfect 1.000 under a sigma half the size of the data. A
  statistic with no room left to be wrong in reporting 1.000 is the most
  confident possible way to say nothing.
- **Trustworthiness runs 0.34 to 0.76 and continuity 0.24 to 0.78**, and 4 to 10
  of the 11 proteins are flagged as having positions that should not be read.
  `k` is clamped to 6 of 11 — more than half the cohort — so "neighborhood" has
  stopped meaning anything local.
- **`tmscore` and `biophys` correlate at Spearman 0.883** over their pairwise
  distances, just under the 0.90 redundancy threshold. The three fused spaces
  split their contribution 50/50 and that split is honest arithmetic, but the
  two blocks are closer to saying the same thing than their names suggest. This
  is the number ADR 0002's `early` warning gestures at and cannot itself
  provide.
- **The censoring rate is exactly 0.000.** Nothing is hidden: at N=11 Foldseek's
  per-query cap of 1000 cannot bind, so there is no fill. On a 2530-protein
  matrix the same field reads 60.5%.
- **Seven of the ten pairs report no cluster ARI.** It is withheld whenever
  either space found fewer than two clusters, because the adjusted Rand index of
  two single-cluster partitions is 1.0 *by convention*. Before that guard, this
  demo printed "families vs fused_late: 1.000" beside a neighborhood Jaccard of
  0.291 — two spaces "agreeing perfectly" when neither had found anything.
- **The negative controls give the number to compare against.** A Leiden fitted
  to a random matrix of the same shape scores a silhouette of 0.105 — not zero,
  because clustering noise produces clusters. `structure` clears it at 0.503.
  `fused_early` does not meaningfully, at 0.176. One of those partitions is a
  finding and one is not, and the picture alone does not tell you which.
- **`fused_early` cannot run its random-distance control at all**: its random
  matrix clusters into a single group, which has no silhouette. It appears under
  `negative_controls.skipped` with that reason rather than vanishing, because a
  shorter list reads as "ran and found nothing".

---

## Where the output lands

```
demo/multispace/output/
├── blocks/{block_id}/          features.npy, protids.txt, manifest.json
├── spaces/{space_id}/
│   ├── embedding_pca_umap.tsv  the 2-D layout
│   ├── clusters.tsv            this space's own Leiden partition
│   ├── faithfulness_*.tsv      trustworthiness and continuity, per protein
│   ├── diagnostics.json        all nine diagnostics
│   └── manifest_*.json
├── coregistration/
│   ├── summary.tsv             one row per pair, four metrics
│   ├── {a}__vs__{b}.tsv        per protein
│   └── index.json              the shared protid index, and what each space lost
├── enrichment/                 FDR-corrected, describes the `structure` partition
└── final_results/              every legacy output, byte-identical
```

Read `docs/INTERPRETING.md` before reading any of it. It is organised as *this
is licensed, that is not, and here is the number that tells you which*.

---

## Things this demo cannot show you

Stated because the alternative is a reader assuming otherwise.

- **Censoring.** At N=11 the per-query cap never binds. The censoring machinery
  is exercised by `parity.synthetic_matrix` at N=750, where 60% of the matrix is
  fill.
- **Reducer determinism above N=500.** Below ~500 proteins scikit-learn's PCA
  picks a different solver and the bug PR #106 fixed cannot occur at all.
  `tests/test_determinism.py` covers it at N=750.
- **A partition worth trusting.** See above — the diagnostics say so.
- **Cross-environment reproducibility of the clustering.** At this N the kNN
  graph is nearly complete and Leiden's optimum is degenerate, so two conda
  environments differing only in their scipy version return different
  two-cluster memberships. At N=250 they agree exactly. This affects the
  pre-existing `leiden_clustering` rule identically.

---

## Trying your own

Every knob is in `config.yml`, annotated. The two most instructive edits:

**Change a fusion strategy.** Set `fused_late.strategy` to `early` and compare
the contribution shares. Under `early` a block's share is essentially its column
count, because standardizing equalizes columns rather than blocks — `tmscore`
takes 73.3% against `biophys`'s 26.7% on eleven proteins, and the warning says
so with the run's own numbers.

**Add a block to a space.** Put `threedi` into `fused_late.blocks` and watch the
redundancy section: three blocks means three pairwise correlations, and the
question of whether any two of them are saying the same thing gets sharper as
you add more.

To add a block that does not exist yet, see `docs/EXTENDING.md` — it takes an
entry point in your own package and no change to this repository.
