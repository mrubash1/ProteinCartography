# Adversarial review log

Findings from the five review gates in `PLAN.md` §4.3, with resolutions.
"Blocks" findings must be resolved before the next commit group starts.

---

## Gate A — after Phase 0.5

**Date:** 2026-08-16
**Target:** *did the claim verification actually verify, or did it pattern-match?*
**Method:** two independent re-derivations, each briefed only with the claim, the
source files, the pinned foldseek binary, and the raw production matrix. Both
were explicitly forbidden from reading `docs/EXPLORATION.md`, `CLAUDE.md`, or
`PLAN.md`, and neither saw the first analysis.

**Outcome: PASS.** Both claims independently confirmed. Every quantitative
result in the first pass reproduced. Six refinements, four of which improve the
build plan. No finding contradicts a conclusion; the disagreements are all
definitional or about wording, and each has been adopted.

### Limitation on independence — recorded honestly

The claim-B reviewer reported that `CLAUDE.md` was **auto-injected into its
context by the harness** before it received the task. It did not open the file,
but it cannot claim zero exposure, and the same injection presumably applied to
the claim-A reviewer. This partially compromises the independence Gate A depends
on.

Assessed as **not determinative**, on evidence rather than assertion:

- `CLAUDE.md` contains no claim-B conclusion at all — it says only "Claim B is
  still unverified" — so there was nothing for that reviewer to anchor on, and it
  produced findings (the AA-vs-3Di measurement, the dbtype rejection) that appear
  nowhere in it.
- `CLAUDE.md` *does* contain claim-A figures, and the reviewer **disagreed with
  two of them** in exactly the way the first pass did, then derived both the
  `CLAUDE.md` value and the corrected value from its own computation. Parroting
  does not produce a reconciliation of two denominators.
- Both reviewers produced substantial material absent from any prior document
  (the mirror-score distribution, the prefilter-vs-cap distinction, the `clust`
  dbtype error, the similarity-type membership experiment).

**Action for Gates B–E:** brief future reviewers in a worktree or with an
explicit instruction that the injected project file is untrusted context for the
purposes of the review. Logged so the weakness is not silently inherited.

### A1 — "unsearched" is the wrong word for the cause · note · adopted

The first pass, following `PLAN.md`, described censored pairs as "never
searched". With `--prefilter-mode` at default, Foldseek **does** examine every
pair at the prefilter stage; it declines to carry more than the top 1000 per
query into TM-alignment.

A zero therefore means *"this pair lost the per-query top-1000 cut"*, a **ranking**
outcome on a competitive list — which is precisely why the same pair can be
reported for query A and dropped for query B, and is the mechanism behind the
53.6% one-way rate.

**Resolution: fixed.** Wording corrected in `docs/EXPLORATION.md` §4.1 and
carried into ADR 0009. Not a change to the conclusion.

### A2 — the distinction survives in the file; `pd.read_csv` destroys it · should-fix · adopted, and it shrinks commit group 4

The first pass established that fills (`0.0`) and measured values (`%.3E`) are
perfectly separable in the file. It did not draw the consequence. Gate A did:
the loss happens at `dim_reduction.py:57`, in `pd.read_csv`, not at write time.

**Therefore the censoring mask is reconstructible from any existing matrix file
by string form during parse — no Foldseek re-run and no `.m8` files needed.**
`PLAN.md` group 4 proposes reconstructing the mask from raw `.m8` alignment
files; that is unnecessary for output this pipeline wrote.

**Resolution: adopted, and the group 4 design changed.** Primary mechanism is
string-form detection inside `matrix_io.py` during parse. The `.m8` route is
retained only as a fallback for matrices this pipeline did not produce. Recorded
in `docs/EXPLORATION.md` §4.1 and ADR 0009. **This removes most of group 4's
projected cost.**

### A3 — the zeros are not merely missing, they are off-distribution · note · adopted, and it is the PR's best number

925,435 censored cells have a measured mirror, so their true value is knowable
from the same file: median **0.772**, with 96.4% above 0.5 and 73.2% above 0.7.
Meanwhile the lowest score Foldseek reports anywhere in the matrix is **0.0549**
and the median reported score is 0.801.

**The reported-score distribution has no low end.** Writing `0.0` does not merely
lose information — it inserts a value Foldseek never produces.

**Resolution: adopted** as the motivating number for `docs/PR_NARRATIVE.md`.

### A4 — "sequence-derived label" overstates claim B · should-fix · adopted

Cluster *membership candidates* come from a 3Di+AA **structural** alignment over
a 3Di k-mer prefilter. Two proteins with high AA identity that never align
structurally never become an edge. AA identity only *weights* edges inside the
greedy set-cover.

**Resolution: fixed.** The characterization is now *"a structurally-gated graph
with sequence-weighted clustering"* throughout. Applies to `docs/EXPLORATION.md`
§4.2, ADR 0003, and `docs/FOLLOWUPS.md` #9.

### A5 — the flag is the weak argument; the dbtype is the strong one · note · adopted

The first pass argued the TM-score DB is never *passed* to `clust`. Gate A showed
`clust` **cannot accept it** — dbtype `0x66` versus the required `0x05`
Alignment, and the binary errors out. Clustering on TM-score is not expressible
with this module, so the conclusion holds even under `--similarity-type 1`.

**Resolution: adopted** as the primary argument in `docs/EXPLORATION.md` §4.2 and
ADR 0003. The flag argument is now secondary.

### A6 — the open sub-question is closed: it is amino-acid identity · resolved

The first pass left the alphabet unresolved and flagged it rather than guessing.
Gate A measured it: extracting AA and 3Di sequences for the 11 demo structures
and walking each alignment's CIGAR, the stored identity matched **amino-acid
identity in 121/121** alignments and 3Di identity in 15/121.

Residual uncertainty stated honestly: Gate A did not read foldseek's source to
prove `clust` reads that specific column. The DB holds only score and seqId,
which is exactly the 1-vs-2 the flag offers. **~95% on the alphabet; 100% on "not
TM-score", which is independent of it.**

**Resolution: closed.** Recorded in `docs/EXPLORATION.md` §4.2.

### A7 — the axes and the colors come from different similarity measures · should-fix · adopted as the headline

Larger than the flag question. The pipeline computes TM-scores and uses them for
the embedding **coordinates**, then **colors** those points with a label whose
edge weights are amino-acid identity — in a rule named
`plot_similarity_strucluster`.

**Resolution: adopted** as the framing for `docs/FOLLOWUPS.md` #9 and the
candidate standalone upstream PR. This is the version that should reach the
maintainers.

### Findings the first pass got right and Gate A confirmed unchanged

Recorded so the pass is auditable in both directions:

| Quantity | Both passes |
|---|---|
| absent-pair fill value | literal string `"0.0"`, `foldseek_clustering.py:244` |
| measured zeros in the file | **0** |
| zero fraction | 60.48% (60.4766%) |
| non-zeros per row | min 983, median 1000, max 1000; 97.8% exactly at 1000 |
| non-zeros per column | min 4, median ~896, max 2328; no cap |
| responsible flag | `--max-seqs`, default 1000 |
| E-value default | 10, **not** 0.001 — not the binding constraint |
| positional diagonal == 1.0 | 2/2530 (0.08%) |
| label-aligned diagonal == 1.0 | 2530/2530 (100%) |
| one-way reported pairs | 36.6% per-cell / **53.6% per-pair** — both derived |
| max abs asymmetry | 0.9944 all-cells / **0.6711** both-reported — both derived |
| `clust` input DB | `foldseek_out`, the alignment DB, never `foldseek_tmscore` |
| `--similarity-type 2` | "sequence identity" per the pinned binary's own help; also its default |
| `--alignment-type` | unset, defaults to 2 (3Di+AA); TM alignment (1) never requested |

### Consequent changes to the plan

| Change | Where |
|---|---|
| Group 4 mask built by string form at parse, not from `.m8` | ADR 0009, group 4 |
| Group 4 cost estimate reduced | `docs/EXPLORATION.md` §11 |
| `struclusters` described as structurally-gated / sequence-weighted | ADR 0003 |
| `representation: direct` needs an explicit symmetrization decision | ADR 0001 |
| Censoring wording is "lost the top-1000 cut", not "never searched" | ADR 0009 |

---

## Gate B — after commit group 2 (abstractions)

**Date:** 2026-08-16
**Targets:** (1) *will this design survive contact with groups 6–8?* — a reviewer
tried to specify five hypothetical blocks against the API and report where it
forces something awkward or impossible. (2) *find an input on which this code
produces a wrong answer.*
**Method:** two independent reviewers, each briefed with the diff and the
contract only, both required to run code rather than speculate. Both were told
to treat `CLAUDE.md`/`PLAN.md` as untrusted context — the mitigation logged
after Gate A.

**Outcome: FAIL, then fixed.** Fifteen findings, **two of them "blocks"**: code
that returned wrong numbers and reported success. Every finding below is fixed
in the commit it belongs to, with a regression test naming it. The two "blocks"
findings were both in the guards that the rest of the design leans on, which is
the argument for having run this gate before four providers were written against
the API rather than after.

**A test that passed for the wrong reason.** `test_freshness_is_content_based`
and `test_freshness_ignores_mtime` were green while the cache could never hit
across processes (B3), because `write_block` mutated the caller's manifest and
the tests handed the same object to both calls. Worth recording on its own: the
suite was not merely failing to catch the bug, it was actively asserting the
broken behavior was correct.

### Blocks

| # | Finding | Fix |
|---|---|---|
| **B1** | `repair=True` with a duplicate column label silently produced **wrong numbers**. `assert_same_label_set` compared sets only, so `[A,B]` vs `[A,B,B]` passed; the reorder then built `{label: position}`, keeping the last duplicate. A 2×3 matrix became 2×2, one column was discarded, and cell (A,B) returned 0.7 where the true value was 0.4 — with `is_aligned` reporting True. Its own error text claims to catch this case. | Duplicate and length checks on both axes, before anything reorders. |
| **B2** | `ProteinIndex.align` silently took the **last duplicate** from the *source* labels. The index rejects duplicates with a paragraph about doubling a protein's weight; the source side, which comes from a data file, had no such check. `align(['A','B','A'], …)` returned source row 2 for A, and the length check still passed. | Duplicate source labels are refused; when a duplicate also masks an absence the error names both. |

### Should-fix, all fixed

| # | Finding | Fix |
|---|---|---|
| B3 | `is_fresh` could **never** return True across processes: `write_block` injected the stored values' digest into `manifest.extra`, which feeds `cache_key`, so only a caller who had already computed the block could match. | Output-derived facts moved to `derived`, excluded from `cache_key`; `write_block` copies rather than mutates. |
| B4 | float32 **underflow** produced a zero that was neither the fill token nor caught by the measured-zero guard — invisible to both checks, on the one file property the loader exists to preserve. | Compared in double against the cast value; underflow and overflow both warn. |
| B5 | `assert_aligned` raised `TypeError`, not `MatrixAlignmentError`, when the label lists agreed positionally but differed in length. Any caller catching the documented exception crashed. | Length mismatch is rejected earlier, with the documented exception. |
| B6 | The `fusable` protection was keyed on the **user-chosen block id**, so it protected `taxonomy:` and missed `tax:`, `Taxonomy:`, `lineage:`. The group-2 commit message claimed the opposite; that claim was false and has been corrected. | Keyed on `provider` first, block id case-folded as a fallback. An explicit `fusable: true` on a known signal now requires a written justification. |
| B7 | `alignment_verified` accepted any truthy value, so the string `"false"` switched the gate **off** — the opposite of what it says, on the gate standing between a user and a 99.92%-wrong read. | Requires a real boolean. |

### Design findings from the API probe, all adopted

| Finding | Fix |
|---|---|
| **The single anonymous `mask` field** had no declared polarity (numpy.ma: True=invalid; pandas/sklearn: True=valid) and no declared meaning. Three future providers each need a per-cell annotation — censoring, absence, confidence — and would have filled one slot with three different ideas. The array is written to disk, so the ambiguity would have been permanent. The reviewer demonstrated a 10%-missing block reporting `censoring_rate = 0.9` with no complaint. **Named as the one change to make before any provider ships.** | Named channels with fixed polarity and dtype; unknown names rejected; `mask=` still accepted as the censoring channel. |
| **Asymmetric pairwise blocks were unrepresentable.** The condensed-triangle shape check made symmetry a type-level assertion, so half of every directed TM measurement had to be discarded with nothing recording that it had been — while ADR 0001 simultaneously required recording a `kept-asymmetric` option that could not exist. | `pairwise_directed` kind, plus a required `symmetrization` on symmetric pairwise blocks. |
| **`features` dtype was unchecked**, so a ragged object array passed construction and failed much later inside `np.save`. | Rejected at construction, with the storage contract in the message. |
| **`with_protids` relabelled rows with no checks** — a permuted list would have attached every protein's values to a different protein, the exact failure the design exists to prevent, offered as a convenience method. | Refuses a different-length or duplicated list; points at `align` for reordering. |
| A NaN with no channel explaining it was accepted, and became a coordinate. | Rejected unless a `censored` or `absent` channel covers it. |
| `values_digest` hashed the pre-cast array while the file on disk was float32, so it could never verify its own file. | Hashes the array as stored. |
| `.tmp` staging did `rmtree(target)` then `os.replace`. | Swaps in first, retires the old directory after. |
| `repair=True` was silently ignored when `require_alignment=False`. | Refused as contradictory. |
| `cap_detected` false-positived on any matrix with a uniform per-row count — e.g. one censored cell per row. | Requires the columns *not* to show the same pile-up, which is what the docstring always claimed. |

### Deferred, with reasons

| Finding | Why deferred |
|---|---|
| No incremental/partial block writes; `is_fresh` is one boolean over a whole block, and `protids_digest` is order-sensitive so a reordered cohort file forces a full recompute. | Real, and it bites the expensive Phase 4 providers (ESM over thousands of proteins), not anything built yet. `store.py` is the only thing that would change. → `docs/FOLLOWUPS.md`. |
| `SpaceSpec.weights` is `dict[block_id, scalar]`, so the per-protein weights ADR 0002 names as `graph`/SNF's unique advantage have nowhere to live. | Phase 5 concern. ADR 0002 overclaims today and is flagged to be corrected when `graph` is implemented. |
| `metric: "jaccard"` is only reachable on a fixed-width float matrix, so a variable-length domain architecture must be flattened to a binary indicator that discards order and multiplicity. | Phase 4 (`domains`) concern. The `pairwise`/`distance_metric` route works meanwhile. |
| `BlockConfig` and `BlockSpec` describe the same concept with different fields and no bridge between them. | The bridge is written in commit group 5, where the first real provider needs it. Noted so it is not forgotten. |
| Frozen config dataclasses wrap mutable dicts, so post-construction mutation bypasses validation. | Configs are built once from YAML; low value, and the fix is noisy. |
| `ProteinCartography.spaces` is shipped but not importable by its dotted path, because modules use the repo's flat-import convention. | Matches existing repo convention, so not a regression. Would need the whole package's import style changed. → `docs/FOLLOWUPS.md`. |

## Gate C — after commit group 5 (the port)

**Date:** 2026-08-16
**Target 1:** *construct an input where the ported path and the legacy path
disagree.*
**Target 2:** *break the parity test itself* — deliberately mutate the pipeline
and confirm the test fails.

**Outcome: PASS, after the first run failed and changed the design.**

The headline is target 2, and the first attempt at it produced the most useful
result of the whole build.

### C1 — the 11-protein fixture is structurally blind to four realistic errors · **blocks** · fixed

The first mutation run scored **4 of 8 detected**. Four survived, and the reason
was not the parity test — it was the fixture:

| mutation | why it could not be seen at N=11 |
|---|---|
| PCA `n_components` 30 → 20 | both clamp to `min(matrix.shape)` = 11, so the two configurations are literally the same computation |
| UMAP `n_neighbors` 80 → 40 | both clamp to `n − 1` = 10 |
| Leiden `n_pcs` 30 → 10 | both clamp to `min(n−1, n_vars−1)` = 10 |
| censoring fill `"0.0"` → `"0.00"` | all 121 pairs are measured, so the fill token is never emitted and cannot differ |

This is `PLAN.md`'s N>500 warning, confirmed by measurement rather than by
argument, and extended: the plan predicted the *PCA solver* problem, and the
same fixture turns out to hide three further parameters and the entire censoring
mechanism. **An end-to-end parity test on the demo fixture is necessary and
demonstrably not sufficient.**

**Resolution: a component-level parity test at N = 750.** Running the whole
pipeline at that size would mean synthesizing 750 PDB files and a Foldseek run,
but the port only touched the reduction step, and that step consumes a matrix
rather than structures. So the matrix is generated directly — seeded and
procedural, so the seed is the fixture and nothing derived from internal data is
committed — and `dim_reduction.py` is run from both checkouts on it.

The generated matrix reproduces the production matrix's measured shape: **60.00%
censoring** against the real 60.48%, rows uniform at exactly the per-query cap
while columns vary (256–347 at N=750), an exact 1.0 label diagonal, Foldseek's
`%.3E` formatting, and no measured zeros. It can also reproduce the PR #106
column permutation on demand.

**At N = 750, 5 of 5 reducer mutations are detected, with no holes** — including
`svd_solver="full"` → `"auto"`, which undoes PR #106's determinism fix and is
invisible at N=11 because the randomized solver only engages above 500 rows.

### C2 — the mutation harness scored a non-applying mutation as a pass · **blocks** · fixed

`pca_solver`'s anchor text had moved into the reducer core during the port, so
the patch never applied — and the harness reported **DETECTED**, because it
caught the resulting exception in the same branch as a pipeline crash.

A mutation suite that scores "nothing was mutated" as a pass is a mutation suite
that will eventually report a perfect score while testing nothing. Outcomes are
now three-valued — `detected` / `survived` / `did not apply` — and the last is
reported separately and fails the run.

### C3 — an ambiguous anchor silently mutated the wrong call site · **should-fix** · fixed

Three of the first reducer mutations survived, and all three were *my* errors
rather than holes in the test:

- `n_components=30` appears **twice** in `main()`, once per branch, and the
  harness replaced only the first — the `pca_tsne` branch, which a `pca_umap`
  run never executes.
- `n_neighbors` and `perplexity` were mutated on the *reducer's* defaults, which
  nothing reads, because `dim_reduction.py` passes both explicitly.
- the t-SNE mutation was run under `pca_umap`, which never reaches t-SNE.

Two of those look identical to a hole in the test from the outside, which is
what makes this worth fixing rather than just correcting. `_patched` now checks
the occurrence count against a declared expectation and replaces *all* matches;
an ambiguous anchor is an error. Mutations also declare which `--mode` exercises
them.

That the report distinguishes "unexplained hole" from "expected to survive" is
what made these three visible as mistakes rather than findings.

### Target 1 — inputs where the two paths could disagree

Checked, no disagreement found:

- **Permuted columns.** `profile` is invariant to a consistent column
  permutation, which is why the shipped UMAP survives pre-#106 output. Asserted
  rather than assumed: `test_profile_distances_are_invariant_to_a_column_permutation`
  builds the same matrix twice, permuted and not, and compares the full pairwise
  distance matrix.
- **N below the UMAP threshold.** The `N < 3` fallback reuses the input's
  existing `PC` columns instead of running PCA on PCA output. I dropped that
  behavior on the first pass at the shim and caught it re-reading the original;
  it would have changed coordinates silently in the one regime nobody inspects.
  Now covered by `test_umap_falls_back_below_three_points` and preserved through
  an explicit `input_column_names` argument.
- **N at the solver boundary.** Covered by the N=750 suite above.
- **A protein absent from the matrix, and duplicate protids.** Both raise rather
  than reindex — `ProteinIndex.align`, and the duplicate guards added at Gate B.

### Result

| suite | scale | outcome |
|---|---|---|
| end-to-end parity | N=11, 4 pipeline runs | 90 files compared, 87 byte-identical, 3 identical after normalizing Plotly's figure uuid, **0 differing** |
| reducer parity | N=750, both modes | byte-identical against the baseline |
| mutation testing | N=11, 200 and 750; 14 mutations | **10 detected, 4 survived as expected, 0 unexplained holes, 0 did not apply** |

`mutation_check.py` exits non-zero on any unexplained hole or non-applying
mutation, so this is a check rather than a report.

The detections include both regressions that matter most: reintroducing PR
#106's unsorted column order, and reverting `svd_solver` to `"auto"`.

### C4 — a claim in the first draft of this entry was wrong · corrected

That draft said "every one of them is covered by a corresponding N=750 mutation
that *is* detected." **That was false for two of the four.** The N=750 suite
runs the *reducer*, which starts from a matrix, so it never exercised:

- `censoring_fill_value` — the fill is written by `pivot_foldseek_results`,
  upstream of any matrix;
- `leiden_n_pcs` — Leiden forks from the matrix independently and the reducer
  suite never touches it.

Both were covered by nothing at all. Two further component runners now close
that: `run_pivot` drives the pivot from a synthetic *raw pair list* (which is
where fills are created), and `run_leiden` drives Leiden on the N=750 matrix.
Both mutations are now detected.

### C5 — the fixture was statistically realistic and biologically empty · fixed

`component_leiden_n_pcs` still survived after being added, and the reason is a
second kind of fixture inadequacy, distinct from size:

**the generated matrix was uniform noise.** Its censoring, cap signature,
diagonal and number formatting all matched production, so it looked realistic —
but it had no cluster structure, and a clustering-parameter mutation therefore
had nothing to bite on. Leiden at 30 principal components and at 10 returned the
*same partition of noise*.

The generator now plants `n_clusters` contiguous groups, within-cluster scores
0.70–0.95 and between-cluster 0.10–0.45 (measured on the fixture: 0.825 vs
0.275). Real TM matrices look like this; noise does not.

**A fixture can be statistically faithful and still test nothing.** Matching the
marginal distributions is not the same as matching the structure the code is
looking for.

### C6 — the same mutation-design error, three times · noted in the harness

`component_leiden_n_pcs` survived once more even after the structure was
planted, because I anchored it on `def scanpy_leiden_cluster(..., n_pcs=30)` —
a default that `main()` always overrides from argparse. The same mistake had
already happened twice in the reducer suite.

Each time it presents as "survived", indistinguishable from a genuine hole. The
`Mutation` docstring now warns about it explicitly: anchor on the value the
executed path actually reads, never on a default the caller passes over.

**Standing consequence.** The four N=11 survivors are recorded in the harness
with `expected_to_survive` text naming the clamp that hides them, so they read as
documented limitations rather than as passes. If anyone later shrinks the N=750
fixture below 500, those annotations are what will explain why the suite
suddenly proves less.

## Group 3 — cohort selection

No gate is scheduled here; group 3 rides on Gate C's parity machinery. Three
findings came out of building it anyway, and two are defects in already-reviewed
code.

### G3.1 — a config key that parsed, validated, and did nothing · **blocks** · fixed

`from_legacy` has two branches. The modern one, taken when a config has
`blocks`/`spaces`, passes the user's `cohort:` mapping through. The legacy one,
taken by **every existing config**, rebuilt that mapping from scratch and copied
only `max_structures` into it. So `cohort: {selection: accession}` in a
plain `config.yml` was silently discarded, and the DAG used the default rule.

The failure mode is the bad one: the key parses, `CohortConfig` validates it,
`_reject_unknown_keys` accepts it, and nothing anywhere reports that it was
dropped. It shipped in commit group 2 and survived Gate B, because every test
exercised the modern branch — `minimal()` in `test_config_schema.py` defines
`blocks` and `spaces`, so the whole cohort section of that file tested the path
almost no user takes.

Found by writing a config with `selection: significance` and noticing the rule
count did not change. Both branches now share `_cohort_from_legacy`, and there is
a regression test on each branch rather than on one.

**Generalization worth carrying: a test helper that always takes the same branch
is a blind spot with a name.** `minimal()` is convenient precisely because it is
the modern shape, which is why nothing tested the legacy shape.

### G3.2 — the additive-output allowance was direction-blind · **blocks** · fixed

`cohort_report.json` is the first file this work adds to the output tree, so
`compare_trees` needed a fourth category beside compared / excluded /
nondeterministic. The first version allowed an `ADDITIVE_OUTPUTS` path to be
missing from the *second* argument, on the assumption that the second argument is
the baseline.

It is not. `test_default_output_is_unchanged_from_the_baseline` calls
`compare_trees(head, base)` and `mutation_check` calls
`compare_trees(reference, mutated)` — the orders are opposite. The real parity
run failed with `only in A: protein_features/cohort_report.json`, which is the
good outcome; the bad one was live in the other call sites, where the allowance
would have excused a **deleted** output.

`compare_trees` now requires `baseline="a"` or `baseline="b"` to be named, with
no default, and the allowance does not apply at all when neither is given — which
is the correct behavior for every self-diff and every mutation run. A test pins
both orders, and another proves no additive pattern can match a `CRITICAL_OUTPUTS`
path.

### G3.4 — `tmalign` mode reuses the column positions for different quantities · **blocks** · guarded

Found while checking G3.3, which turned out to be wrong (see below).

`foldseek_apiquery.py` accepts `--mode tmalign`. A live query against
`afdb-swissprot` with the demo actin structure returns the **same 21 columns in
the same positions** as `3diaa`, with different meanings:

| column | 3diaa (recorded fixture) | tmalign (live, 938 hits) |
|---|---|---|
| `evalue` | `1.647e-79`, spans 79 orders | `0.402 … 0.9999`, all in [0, 1] |
| `bits` | up to 3274 | 22–99, ≈ TM×100 (r = 0.93) |

The column named `evalue` holds a **TM-score**. Nothing renames, nothing errors,
and the mode is recorded nowhere in the output — so the polarity silently
inverts for any consumer.

Two consequences, one pre-existing and one mine:

- `extract_foldseek_hits.py` filters `evalue < 0.01`. On that live response it
  keeps **0 of 938 hits**. → `docs/FOLLOWUPS.md` #25.
- `hit_significance.py`, written two commits earlier, ranks `evalue` ascending.
  It would have put the unrelated GTP pyrophosphatase (TM 0.402) above actin
  (TM 0.9999) — the exact inversion `SIGNIFICANCE_MEASURES` exists to prevent,
  arriving through the *data* rather than through the code.

`hit_significance.py` now refuses tmalign-shaped input instead of guessing:
values bounded in [0, 1], never small, *and* bit scores at TM-score scale. Two
conditions rather than one because a weak 3Di-AA search really can return only
e-values near 1; requiring TM-shaped bit scores alongside makes a false positive
very unlikely, and the check errs toward not firing. Verified against both real
files: it fires on the live tmalign response and stays quiet on the recorded
3diaa fixture (810 accessions scored).

**Refusing is deliberate and temporary.** The correct fix is to record the mode
next to the results, after which either mode can be read with confidence. That
is in PLAN.md as a scoped exploration, not done here, because making the mode
configurable changes search behavior and therefore the cohort.

**The generalization is the value here.** The project already has the rule
*never index a labeled matrix positionally* (ADR 0007). This is the same defect
one level up: **a column name is not a contract when its meaning depends on a
run mode that is not recorded.** ADR 0007 protects against columns moving. This
is columns staying exactly where they are and meaning something else.

### G3.3 — ADR 0008 asked for a measurement that cannot exist · **the correction was itself wrong** · re-corrected

The ADR specified `significance_rule: {foldseek: best_tmscore}`. I corrected it
to say that a TM-score is unobtainable, because the pipeline's TM-scores come
from the local all-versus-all run, which operates on the downloaded structures —
so ranking the cohort by TM-score would need the structures the ranking exists to
choose. I called it "not a DAG-ordering problem" and "no implementation resolves
it", and committed that in `1ac5dcd`.

**That was an overclaim, and the argument had a hole in it.** The circularity is
real for the **all-versus-all matrix**. Cohort ranking does not need the matrix;
it needs one score per candidate against the queries, which is a different and
much cheaper object. Checking `--mode tmalign` — which I had read past in
`foldseek_apiquery.py` before writing the correction — showed the web API returns
exactly that, before anything is downloaded.

What generalized from one verified fact ("the recorded 3diaa fixture has no
TM-score column") to a claim about the whole API ("the web API reports no
TM-score") was an inference, not a measurement, and it was stated with the same
confidence as the measurement. **The tell was available: I had already read the
`SET_MODES = ["3diaa", "tmalign"]` line in the same file.**

The ADR is re-corrected. The decision does not change — the default measure is
still the e-value — but the reason does, and the new reason is more useful than
the old one: TM-score ranking is *feasible* and blocked by an unrecorded mode
plus a column-semantics collision (G3.4), not by a law of the DAG. That turns a
dead end into a scoped piece of work, now in PLAN.md.

The honest consequence stands either way: an e-value is alignment significance,
not structural similarity, so `significance` as shipped selects
**confidently-detected** hits rather than structurally close ones. Weaker than
the ADR promised, and still the only rule of the three that is both reproducible
and principled.

### What the demo fixture turned out to show

Not a finding, but the most useful number group 3 produced. On the repo's own
11-protein search-mode fixture: 24 hits, 20 surviving the metadata filter, 10
admitted by `max_structures`. Half the candidates dropped, and the discarded half
is taxonomically different from the kept half — every Chiroptera hit discarded
(0% retained vs 40% discarded), 4 of 5 Artiodactyla kept.

The demo the project ships to explain itself already exhibits the bias ADR 0008
is about, and no file recorded it before this group.

### Mutation testing after group 3

17 mutations: **12 detected, 5 survived-as-expected, 0 unexplained holes.**

Two new mutations, both detected: the default rule quietly sorting (the exact
change ADR 0008 rejected), and an off-by-one at the truncation point.

One new deliberate hole, recorded rather than closed. Inverting the significance
polarity survives, because the default config never executes that path — and
making it do so would mean changing the default cohort, which is the one thing
this work promises not to do. It is covered by unit tests that assert the e-value
and TM-score directions against each other. Recording it is the point: the
parity test cannot see anything the default configuration does not run, and that
limit should be written down rather than discovered later.

## Gate D — after commit group 8 (fusion and diagnostics)

*Not yet run.*

## Gate E — before opening the PR

*Not yet run.*
