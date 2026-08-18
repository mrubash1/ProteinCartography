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

## Group 6 — free blocks

### G6.1 — the block and space rules had never run · **blocks** · fixed

`compute_block` and `reduce_space` were wired into the Snakefile in `e72f87d`.
Building the 3Di block was the first time either was *executed*, and both failed
at import:

```
ModuleNotFoundError: No module named 'yaml'
```

They run in `envs/analysis.yml`, which has no PyYAML. `tmscore` failed the same
way, so **no block had ever been computed by the pipeline**. The wiring commit
was verified by `snakemake -n` and by the cluster DAG staying at 16 rules — both
of which passed, and neither of which executes anything.

The fix is not to add PyYAML to `envs/analysis.yml`. That changes the env hash,
which forces a fresh solve of the one environment whose package versions decide
the pipeline's numeric output — the drift this repository has already been bitten
by once, when a fresh solve pulled a numpy-2-built matplotlib in beside a pinned
numpy 1.23.5. A `multispace_config` rule writes JSON instead, and `config_io`
imports PyYAML lazily for the by-hand path.

**Standing consequence: a rule can resolve in the DAG, pass `-n`, and never have
run.** Every remaining group adds rules. Each one needs an actual execution
before its commit claims anything, and "the DAG is unchanged" is not that claim.

### G6.2 — `write_block` discarded the manifest it was handed · **blocks** · fixed

Found within minutes of G6.1, because the first successfully written block had
an empty `extra`.

`write_block(result)` rebuilt a minimal manifest from `result.spec` and ignored
`result.manifest`. Everything a provider records about what it computed *from*
was dropped at the moment of writing: input digests, seed, and provider stats.
`tmscore`'s censoring summary — the number ADR 0009 is built on — had never
reached disk.

Not only lost provenance. `inputs` feeds `cache_key`, so **a changed input matrix
produced an unchanged key and a stale block looked fresh.** Both properties are
now tests.

This is the third finding in this work about the same shape of mistake: an
object is constructed carefully and then quietly replaced by a cheaper one
downstream. Gate B's B3 was the same defect in the other direction — `write_block`
*mutating* the caller's manifest, which made freshness answer itself.

### G6.3 — the descriptor format hides a sequence/structure collision · designed against

`foldseek structureto3didescriptor` emits four fields. Fields 2 and 3 are the
amino-acid sequence and the 3Di string, and they are **both uppercase letter
strings of exactly the same length**. Reading the wrong one yields a *sequence*
k-mer profile labelled as a *structural* one, with nothing in the output to
indicate it — and the whole point of the block is that it is not a sequence
measure.

Same shape as the tmalign collision in G3.4, met a second time in three commits.
The reader asserts what it can (four fields, equal lengths, sequences not
identical) and names the column index as a constant rather than inlining it.

Worth stating as a pattern, since it has now come up twice in one session:
**where two adjacent columns have the same type and different meanings, the type
system is not going to help and neither is a name.** The only defences are an
assertion that would fail if they were swapped, and a test that swaps them.

### Verification at the time of G6.1–G6.3, since G6.1 is about verification

*Superseded by "Verification, group 6 complete" below; kept because these are the
numbers the `threedi` commit was checked against.*

- The shipped `demo/multispace/config.yml` ran to completion: 19/19 steps, two
  co-registered spaces over one protein index, both block manifests populated.
- The 3Di block was checked against real foldseek output rather than a fixture:
  11 demo structures, 3Di lengths matching sequence lengths exactly, pairwise
  distances spanning 0.020–0.118 with the longest, most divergent structure the
  outlier in both farthest pairs. Non-degenerate, and it disagrees with TM-score.
- Parity: 31 slow tests, 0 differing. Cluster DAG 16.

### G6.4 — every provider default was dead code · found by adding a block that disagreed

`compute_block` prepared the provider's parameters like this:

    params.setdefault("normalization", block.normalization)

and `BlockConfig.normalization` defaulted to `"unit_mean_distance"`, unconditionally.
It was therefore never `None`, the `setdefault` always won, and every provider's own
`params.get("normalization", ...)` default was unreachable code. Both providers that
existed wanted `unit_mean_distance`, so the config default and the provider defaults
agreed by coincidence, and the mechanism looked like it worked.

`biophys` broke the coincidence. Its columns are a pH beside a per-residue charge
beside a dimensionless fraction — pI spans 4 to 12, charge per residue spans about
−0.1 to 0.1 — so the unnormalized euclidean distance between two proteins is the
isoelectric point and essentially nothing else. The provider asks for `zscore_within`.
It was getting `unit_mean_distance`, and that value was written into the block manifest,
which is the record a later reader would trust.

**How it got past a test that was written to catch exactly this.**
`test_the_block_standardizes_its_columns_by_default` existed, asserted `zscore_within`,
and passed — because it calls the provider directly, and the defect is in the caller.
This is the same failure as a mutation anchored on a default the caller overrides, which
this work has already recorded twice: *a default is dead code if its caller always
passes the parameter, and no test of the callee can tell you so.* The fix therefore came
with `test_compute_block.py`, driving `main()` through an argv, which is the path the
Snakefile takes. The entry point had had no tests at all.

`normalization` is now optional; `None` means "ask the provider". Nothing else moves:
both existing providers default to the same value they were being handed, `from_legacy`
sets it explicitly, and `to_spec` — which builds a spec with no provider to ask — falls
back to the historical default. The field is still validated when given.

### G6.5 — a transcription error that every derived test tolerated · caught by comparing tables

`biophys` computes hydropathy, charge, isoelectric point and molecular weight from
published constants rather than from Biopython, because `compute_block` runs in
`envs/analysis.yml`, Biopython is not in it, and adding it would change that
environment's hash and force a fresh solve of the environment whose pins exist because a
fresh solve once installed a numpy-2-built matplotlib beside `numpy=1.23.5` (ADR 0006).

The constants are Biopython's, so the two can be compared. That was not decoration:

- The first draft used the **EMBOSS** pKa set (Nterm 9.69, K 10.5, R 12.4, Cterm 2.34,
  D 3.86, E 4.25, C 8.33) rather than the **Bjellqvist** set Biopython uses (Nterm 7.5,
  K 10.0, R 12.0, Cterm 3.55, D 4.05, E 4.45, C 9.0). Both are real, published, widely
  cited tables calibrated against different experiments. Every charge and every pI was
  wrong by a plausible amount — net charge off by up to 1.6 units, pI by up to 0.49.
- Arginine's average mass was transcribed as 174.2017 against Biopython's 174.201. That
  difference is **below the tolerance of the derived molecular-weight check**, which
  passed throughout.

The lesson is about which test to write. Sixteen derived-value comparisons failed at
once and none of them said *which* of fourteen constants was responsible; comparing the
**tables themselves** named it immediately, and was the only check that saw the arginine
slip at all. `test_the_pka_tables_are_biopythons` and `test_the_weight_table_is_
biopythons` are now the first cross-checks in the file. Both skip when Biopython is
absent, which it is in the environment the block must work in.

After the correction: 28 of 28 cross-checks agree — charge to machine precision, pI to
2×10⁻⁵, GRAVY and aromaticity and MW exact.

### G6.6 — intensive versus extensive, and why the default descriptor set is short

Molecular weight is about 110 Da per residue and nothing else. A space that includes it
is partly a map of protein length, which is the exact argument ADR 0003 uses to keep
pLDDT out of any geometry — and unlike pLDDT, nothing about the name `molecular_weight`
warns you.

Rather than bar it, the descriptor table carries an `intensive` flag per descriptor and
the default set is the intensive ones only: mean hydropathy, aromatic fraction,
isoelectric point, charge per residue. `molecular_weight` and `length` remain available;
asking for either is recorded in the manifest as `length_proportional_descriptors` and
warned about on stderr, so the choice is visible to whoever reads the map rather than
buried in a config file. `test_an_intensive_descriptor_does_not_track_length_and_an_
extensive_one_does` measures the distinction instead of asserting it.

### G6.7 — `jaccard` refused rather than silently ignored

`spaces.base.METRICS` contains `jaccard`, and it is the natural distance between two
sets, so it is the obvious choice for `domains`' binary presence vectors. But
`reduce_space` never consults `spec.metric`: it feeds `block.features` straight into a
euclidean PCA. Declaring `metric: jaccard` would therefore have written a claim into the
manifest that no code in the repo honors — a metric that exists only as a label.

The block declares `euclidean`, which is what actually happens, and refuses `jaccard`
with an error explaining that the reducer core is not metric-aware and that euclidean
distance on binary vectors is the square root of the number of families two proteins
differ on. That is a real distance; it simply weights a heavily annotated protein more
than Jaccard would.

This is the third time in group 6 that the honest move was to make an unhonored setting
loud rather than to accept it. It is the same category as G6.1 (rules that had never
executed) and G6.4 (a parameter that was always overridden): **a setting nothing reads
is worse than a setting that does not exist**, because it reads as a decision that was
made.

### G6.8 — where the mutation rule stops, stated rather than eroded

The standing rule is that any new numeric component gets a mutation entry. Neither
`threedi` (previous commit) nor `biophys`/`domains` has one, and that is deliberate
rather than an omission accumulating quietly.

The mutation harness measures what the **parity test** can see, and the parity test runs
the **default** configuration. None of these three blocks is in it — they are reachable
only from a config that defines `spaces`. A mutation planted in any of them would be
reported as "survived", with the reason "the default configuration never executes this
path" — which is already recorded once, against `cohort_significance_polarity`, and
adding three more identical entries would grow the expected-survivor list without adding
information.

They are covered by unit tests instead: 46 for `biophys`, 35 for `domains`, 39 for
`threedi`, plus 12 for the entry point that assembles them. The boundary worth
remembering is unchanged and is the one §0.1 states: **the parity test cannot see
anything the default configuration does not run.** When a block enters the default
config, it gets a mutation entry.

### Verification, group 6 complete

- `demo/multispace/config.yml` runs to completion: **23/23 steps**, four co-registered
  geometries over one protein index — `structure` (tmscore), `local_structure` (threedi),
  `physicochemistry` (biophys), `families` (domains). All four block manifests populated.
- `biophys` checked against real data, not a fixture: on the demo's eleven actins it
  reproduces human β-actin's published pI of 5.29 exactly, and isolates the single
  outlier — a 507-residue fungal actin–histone fusion — at pI 8.7 with positive net
  charge where the other ten sit at −0.03 per residue.
- `domains` on the same cohort finds PF00022 (actin) in all eleven and PF00125 (histone)
  in that same fusion protein, 100% annotated. The resulting space is nearly degenerate,
  which is the correct answer for a cohort this homogeneous.
- ADR 0006's four free blocks — `tmscore`, `threedi`, `biophys`, `domains` — now all
  exist and all import with only numpy and pandas, enforced by
  `test_optional_dependencies.py`.
- Rebased onto `upstream/main` at `36a38c7`; all four carried cherry-picks dropped, so
  the PR diff is now purely this work. Parity re-run against that new baseline: 31 slow
  tests, 0 differing files. Mutation harness exits 0 — no unexplained holes across its
  17 mutations. Unit suite 500 passed / 40 skipped; cluster DAG 16, search DAG 25, both
  unchanged from the baseline.

## Group 7 — co-registration

### G7.1 — the co-registered spaces were never checked to be over the same proteins · **blocks** · fixed

Recorded as FOLLOWUPS #30 during group 6 and fixed here, because it is the precondition
for everything else in this group rather than a defect group 7 happened to notice.

The four blocks draw their protid sets from three different files — `tmscore` from the
similarity matrix, `threedi` from the 3Di descriptor table, `biophys` and `domains` from
the UniProt features table — written by different rules at different points in a run.

The failure mode has no symptom. Two spaces over sets differing by a handful of proteins
reduce cleanly, plot cleanly, and produce a per-protein comparison silently conditioned
on the overlap. Shapes match, so nothing downstream can notice.

`coregistration.shared_index` intersects in a named reference space's order and
**enumerates** every protein each space lost, rather than counting them. A count says
something happened; the list says whether it matters. Intersecting rather than refusing
is the deliberate half — a provider legitimately has no data for a protein — and is
argued in ADR 0011 §1.

### G7.2 — the boundary-tie counter earned itself on the first real run

Neighborhood Jaccard breaks a k-th-neighbor tie by protein index order: reproducible,
and arbitrary. The count of such ties is reported beside every score.

That looked like defensive decoration until the demo ran. The `families` space has
**two distinct points across eleven proteins** — ten actins carry one Pfam family and
the fusion carries two — so all eleven rows tie, and its Jaccard of 0.18 against
`structure` is largely a measurement of protein index order. Read alone, 0.18 is a weak
biological signal. `boundary_ties_b = 11` is what says it is not a biological signal at
all.

The same run surfaced a second one: **four of the eleven demo proteins have
byte-identical 375-residue sequences** under four accessions (Q6QAQ1, P60709, D7RIF5,
P60713), so every sequence-derived space is degenerate on them too. Checked against the
features table rather than assumed to be a float32 storage artifact — the sequences are
identical, so this is real conservation, not a fixture defect.

**The transferable form: a diagnostic that can never fire is decoration, and one that
always fires is noise. This one fires on the exact cohorts where the score it qualifies
is meaningless.** `test_no_boundary_ties_are_reported_when_every_distance_is_distinct`
guards the second half.

### G7.3 — a normalization declared everywhere and applied nowhere · recorded, not fixed

Found while deciding what geometry the comparison should measure. `spec.normalization`
is validated on every `BlockSpec`, written into every manifest on disk, and read by
nothing: `reduce_space` feeds `block.features` straight into PCA. Exactly the shape of
#29 for `spec.metric`, and found the same way — by needing the value and discovering
nothing consumed it.

It matters most for `biophys`, whose four columns are a pH beside a per-residue charge.
Unnormalized, the euclidean distance between two of its rows is the isoelectric point
and almost nothing else.

**Deliberately not fixed here.** Applying it would change the default map, which the
byte-identical requirement forbids. Applying it *only* in co-registration would be
worse: the disagreement metrics would then describe a geometry no map is drawn from, and
nothing on either the metric or the plot would say so. So co-registration shares the
defect on purpose, and states it in `geometry_caveats` on every comparison and in the
manifest. FOLLOWUPS #32; ADR 0011 §4.

This is the third instance of one pattern in two groups — a recorded field that no
caller consults (#29, #32) or a default no caller can reach (G6.4). **A value written to
a manifest is not thereby honored, and a manifest is where that goes unnoticed longest,
because it looks like evidence.**

### G7.4 — the claim B audit found no circular comparison

PLAN.md §Phase 6 required, on claim B being confirmed, an audit of every comparison
using `struclusters` as a structural reference: a "structure versus sequence" contrast
built on a sequence-derived label is circular.

Audited, and the answer is that there is no such comparison to fix. `struclusters`
appears in three places — the aggregated features table, `plot_similarity_strucluster`,
and the fusion path. The first two use it as an **overlay**, which ADR 0003 explicitly
permits: looking at a layout colored by it is fine, letting it move the points is not.
The third is already blocked by `NOT_FUSABLE_REASONS["struclusters"]`, whose text is the
circularity argument.

The one live problem is naming, not circularity — the repo presents amino-acid identity
as structural clustering — and that is FOLLOWUPS #9, a standalone upstream PR.

Recording a **negative** audit result matters as much as a positive one: without this
entry the next reader re-derives it, and the plan's instruction to audit stays
permanently open.

### G7.5 — cross-checked against scipy rather than merely plausible

Spearman and Procrustes are reimplemented in numpy, because ADR 0006 requires the
default configuration to run with neither scipy nor scikit-learn installed. A
reimplemented statistic that is subtly wrong is worse than a missing one; it is
plausible.

Both were checked against `scipy.stats.spearmanr` and `scipy.spatial.procrustes` in an
environment that has them, and agree to **3.3e-16** over twenty random cases. The
Spearman check includes a **60% censored** matrix, because that is the production
censoring rate (ADR 0009) and it makes tie handling the common case rather than an edge
case: ranking tied zeros by position would manufacture an ordering out of file order,
and two spaces would then correlate on that artifact.

Both checks are in the suite behind `importorskip`, so they run wherever scipy happens
to be installed and skip in the bare environment — the same opportunistic pattern as the
Biopython cross-checks in `test_biophys_block.py`.

### G7.6 — why the new metrics get no mutation entry, stated rather than skipped

PLAN §0.1 sets a standing rule: **any new numeric component gets a mutation entry**, and
if it survives, work through three causes in order — fixture too small, fixture lacks
the structure the component reads, or the mutation is anchored where the code never
executes.

Group 7 adds three numeric components (neighborhood Jaccard, rank correlation,
Procrustes disparity) and none of them gets an entry. The reason is a **fourth cause not
on that list**, and it is structural rather than a hole:

> The mutation harness measures whether the **parity test** notices a change. The parity
> test compares this branch's output against the baseline's. Co-registration output does
> not exist in the baseline — it is additive, which is the whole point — so no mutation
> to `coregistration.py` can ever be detected, no matter how large. Every entry would
> survive, and each would be indistinguishable in the report from a real hole.

Adding them would therefore make the harness *worse*: five survivals with reasons
attached is a report someone eventually stops reading.

This is the same shape as the one deliberate survivor already recorded in §0.1 — the
significance-polarity inversion survives because the default config never executes that
path — and it generalizes to a limit worth stating plainly: **the parity test cannot see
anything the baseline does not produce, and everything additive is by construction
invisible to it.** Additive work has to be covered by direct tests instead, which is why
the three metrics carry 33 unit tests and are cross-checked against scipy (G7.5) rather
than being defended by the harness.

### Verification, group 7

- Unit suite: 567 passed, 42 skipped. The two extra skips are the scipy cross-checks,
  which is ADR 0006 behaving as designed.
- **Parity: 33 slow tests, 0 differing files**, against `upstream/main` at `36a38c7`.
  Up from 31 because `test_parity.py` gained two non-slow tests of `baseline_commit`.
- **Mutation harness exits 0** — no unexplained holes. See G7.6 for why group 7's own
  components are deliberately not in it.
- Every one of the five commits verified alone in a detached worktree:
  502 / 519 / 551 / 567 / 567 pass, lint clean, cluster DAG 16 at every point.
- Lint: `ruff check`, `ruff format --check`, `snakefmt --check` all clean.
- Cluster DAG 16 and search DAG 25, both unchanged. Multispace demo 23 → 24, the one new
  rule, and it runs 24/24 to completion.
- Co-registration is opt-in twice: unreachable without `spaces`, and unreachable again
  unless `coregistration.compare` names two of them.
- Pre-existing files touched rises 9 → 10. `calculate_concordance.py` gains a module
  docstring and no behavior change, because PLAN.md requires the superseded metric be
  kept and marked rather than removed (invariant I6).

## Group 7, second half — cluster enrichment

### G7.7 — the fixture came before the statistic, and it was the right order

PLAN §0.5 note 3 said to build the fixture first, because the demo cohort is degenerate
for this: eleven proteins, four with byte-identical sequences, and a `Pfam` column with
two distinct values. Following it produced a 400-protein generated cohort with planted
signal — named terms in named clusters at a stated marginal rate, named columns shifted
by a stated effect size — and its own test file checking those claims by measurement
rather than by reading.

That test file earned itself twice before any statistic existed. It caught the planted
cluster labels not matching the generated ones (`leiden_clustering.py` pads a label to
the digit count of the largest index, so eight clusters are `LC0`..`LC7` and twelve are
`LC00`..`LC11`; the planted labels had been written two-digit), and it caught a recorded
`rate_inside` that was conditional rather than marginal, because the generator drew
`Eukaryota` first and `Chordata` beneath it. Both would have presented later as "the
statistic missed the signal".

The half that matters more is the **null** half. The fixture asserts that `Organism` is
independent of the clustering and that an unplanted family is not concentrated anywhere,
because a statistic that calls everything significant passes every test that only looks
for the planted signal.

### G7.8 — the entry point crashed on a table shape no unit test had built · **blocks** · fixed

Thirty-six unit tests passed and the demo run failed immediately:
`ValueError: cannot insert LeidenCluster, already exists`.

`aggregated_features.tsv` is built by joining `leiden_features.tsv` into everything else,
so **both** of the entry point's inputs carry the cluster column. The test fixture had
written the annotation table with that column dropped, which is a shape the pipeline
never produces.

This is group 6's lesson for the third time — *test the caller's path, not only the
callee's* — and the second time in this work that running a thing end to end found what
a full unit suite could not (the first was `compute_block` and `reduce_space` never
having executed at all). The fixture now writes the pipeline's shape, and the reconciled
case is a test in both directions: the duplicate column is dropped in favour of the
cluster table, and the two *disagreeing* is an error, because that means the tables
describe different runs and an enrichment across them would be wrong in no visible way.

### G7.9 — what the demo actually found, and why it is not a biological result

The demo's two Leiden clusters — seven canonical 375–385 aa actins, four longer ones up
to 507 aa — separate **perfectly** on `pdb_confidence` (max of one cluster 91.83, min of
the other 92.11) and almost perfectly on `Length`, in opposite directions. Across the
cohort the two correlate at **r = −0.86**.

So the enrichment table rediscovers, from the other end, exactly the confound
`NOT_FUSABLE_REASONS["pdb_confidence"]` already refuses to fuse: prediction confidence
tracks length, so a map that let it move the points would make protein length an axis.
Arriving at the same place from an independent direction is the most useful thing the
demo does here, and it is recorded in `demo/multispace/config.yml` so a reader of the
output meets the explanation rather than the number.

Nothing categorical is significant, which is the honest null half: two distinct `Pfam`
values, and most of the lineage vocabulary carried by a single protein.

### G7.10 — the annotated universe shrinks, and the file format is why

For a categorical column the universe is the proteins whose cell is present. A protein
whose annotation was never determined is not evidence that it lacks a term, and counting
it as one biases every term toward whichever cluster is better annotated — annotation
completeness correlates with taxonomy, which is one of the things being tested.

Measured on the 400-protein fixture, that is not a small correction. A protein with no
Pfam families is written as an empty field and read back as NaN, so **after a round-trip
through TSV "carries no families" and "was never annotated" are the same bytes**: 168 of
400 rows, taking Pfam's universe to 232 and raising every fold enrichment accordingly.

The choice stands — an enrichment background is conventionally the annotated background
— but it is made visible three ways rather than argued once: `n_universe` on every row,
the per-column universe in the manifest, and a stderr line from any column that lost
proteins. ADR 0012 §3.

### G7.11 — two defects recorded, not fixed

**`remove_nans` invents a zero** (FOLLOWUPS #34). `plot_cluster_distributions.remove_nans`
has `default=(0,)`, which fires whenever a cluster's values for a column are all missing.
That cluster is then Mann-Whitney tested as a distribution of one synthetic `0.0` against
every other cluster's real values, and marked on the plot. A cluster whose structures all
failed to download reads as significantly low confidence rather than as unmeasured. Same
family as #29 and #32: a value that is present, looks like evidence, and is not. Not
fixed here because it changes a shipped SVG, which the parity requirement forbids;
`enrichment` takes the other branch and reports the comparison as untested with a reason.

**Two of PLAN's four enrichment categories have no data source** (FOLLOWUPS #35).
`fetch_uniprot_metadata` requests neither `ec` nor `cc_subcellular_location`, so EC and
localization have no column. Adding a field changes the columns of `uniprot_features.tsv`
and therefore of `aggregated_features.tsv` — exactly what the byte-identical guarantee
forbids. Enrichment is column-driven rather than category-driven for this reason, and a
requested column that is absent is named on stderr and in the manifest rather than
skipped, because "no enrichment for localization" and "localization was never in the
table" are different facts.

### G7.12 — the scipy cross-check was executed this time, not only asserted

G7.5 established the pattern. Applying it here surfaced something about the pattern
itself: `cartography_tidy` has no scipy, and the environment with scipy that group 7 used
for its manual check has no pytest — so an `importorskip` test can be written, be
correct, and never once run.

All four statistics were checked twice. First directly, against scipy 1.15.2:
Mann-Whitney to 2.2e-16 over 300 cases including heavily tied and 60%-censored input, the
hypergeometric to a **relative** 5.5e-12 down to p = 1e-94, Fisher's one-sided exact test
to 7.8e-15, Benjamini–Hochberg to 2.2e-16. Then as the suite, in
`.snakemake/conda/21cb44d…` which carries both scipy 1.13.1 and pytest: **95 passed, 0
skipped**, against 78 passed / 17 skipped in the bare environment. The 17 are the gated
cross-checks, and they are now known to pass rather than assumed to.

The relative check is the one worth keeping. An absolute tolerance of 1e-12 is vacuous on
a p-value of 1e-30, and the top of an enrichment table is entirely made of those.

### Verification, group 7 second half

- Unit suite: 714 passed, 59 skipped. The 17 extra skips over group 7 are the scipy
  cross-checks; in an environment with scipy the same file is 95 passed / 0 skipped.
- **Parity: 33 slow tests, 0 differing files**, against `upstream/main` at `36a38c7`.
  Unchanged from group 7a, which is the expected result: the default configuration sets
  no `enrichment` key, so none of this executes in a parity run.
- Each of the three code commits verified alone in a detached worktree:
  596 / 675 / 713 pass, lint clean, cluster DAG 16 at every point.
- Lint: `ruff check`, `ruff format --check`, `snakefmt --check` all clean.
- Cluster DAG 16 and search DAG 25, both unchanged. Multispace demo 24 → 25, the one new
  rule, and it runs 25/25 to completion.
- Enrichment is opt-in on its own key, and gated on `enrichment` rather than on `spaces`,
  because a legacy cluster-mode run already produces both tables it needs. A config that
  names no annotation column gets no rule.
- **Mutation harness exits 0** — 11 mutations, 10 detected, one survivor, and that one is
  the `cohort_significance_polarity` survival already recorded in §0.1 with its reason.
  No new entries were added, for the reason recorded in G7.6: the default configuration
  sets no `enrichment` key, so the parity test cannot see this output at all and every
  mutation to it would survive for a structural reason rather than a real hole. The tree
  was checked restored afterwards — no diff against HEAD in any source file.
- Pre-existing files touched stays at 10. The only pre-existing file this half edits is
  `Snakefile`, which was already in the set; `config_schema.py` and
  `demo/multispace/config.yml` are files this work created. `plot_cluster_distributions.py`
  is deliberately untouched (invariant I6) — its defect is FOLLOWUPS #34, not a drive-by
  fix.

## Group 8, first half — fusion

Phase 5 lists four fusion strategies and nine diagnostics. That is too much for one
commit group, and the two halves are not symmetric: the diagnostics are independent of
each other, fusion is not independent of them, and Phase 5's own item 2 (contribution
share) has no meaning until fusion exists. So fusion first, and the diagnostics are their
own group.

### G8.1 — the fixture came before the strategy, again, and again it paid

Group 7b's lesson repeated (G7.7). Fusion's failure mode is not a crash; it is a map that
looks fine and is one block's map wearing several blocks' labels. Neither a shape
assertion nor the eleven-protein demo can see that.

`tests/fusion_cohort.py` plants **two crossed partitions in two different blocks** —
`fold` visible only in a 200-column block at scale 10, `chemistry` only in a 4-column
block at scale 0.01 — with the crossing exact, so all twelve cells hold 20 proteins and
neither partition carries information about the other. That last property is what makes
"this fusion recovered `fold` and stayed blind to `chemistry`" a statement about the
fusion rather than about the draw, and it is why the crossing is built as a repeated
cross-product and permuted rather than drawn independently.

Measured on the generated cohort, by its own tests: separations 3.43 and 3.71 for the two
blocks on their own partitions, 0.99 both ways for the partitions they should be blind to,
and a **7523x** gap in mean pairwise distance between the two blocks. Comparable
information, incommensurate units — which is the case ADR 0002 exists for, made extreme
enough that a broken normalization cannot hide in sampling noise.

The fixture's measuring stick (a between-over-within distance ratio) is implemented by
explicit broadcasting rather than through the Gram identity production code uses. A
measuring stick that shares an implementation with the thing it measures cannot catch that
implementation being wrong.

### G8.2 — ADR 0002's contribution-share formula cannot report anything · **ADR corrected**

The formula is `share_i = w_i · mean(d̃_i)² / Σ_j w_j · mean(d̃_j)²`, and the normalization
contract makes every `mean(d̃)` exactly 1 by construction. So it returns the normalized
weight vector. It is not wrong; it is empty. "Contribution share is a first-class,
computed, recorded output" was meant to mean something a config could not have told you.

What a block actually puts into a fused squared distance is `w_i · mean(d̃_i²)`, and
`mean(d̃²) = 1 + var(d̃)`. Both are now computed. `share` keeps ADR 0002's formula, computed
from the *measured* means rather than from the weights so that it stops being the weight
vector the moment normalization breaks; `realized_share` is the fraction of the fused
quantity the block accounts for, and it is what the dominance warning keys on.

They differ by enough to matter. On the demo cohort, `late` over `tmscore` and `biophys`
at equal weights: nominal **50/50**, realized **34/66** — the other way round.
`biophys`'s normalized distances are far more dispersed across these eleven proteins.

ADR 0002 is amended in place with a pointer to ADR 0013 §3.

### G8.3 — the published SNF kernel is not scale-invariant

Wang et al.'s scaled exponential kernel takes `exp(-d² / (μ·ε))` where `ε` is an average
of local distances. `d²/ε` has units of distance, so the whole exponent scales with the
block: multiply a block by 1000 and its affinities change entirely.

This is not a defect in the paper — SNF is normally applied to one dataset at a time — but
it is fatal to a fusion contract whose first clause is that block scale must not decide
anything. Normalizing to unit mean distance before the kernel, which ADR 0002 requires
regardless, removes it. Measured against the fixture's `narrow_rescaled` block, which is
`narrow` times 1000: `late` reproduces the geometry to a relative **1.1e-15** and `graph`
to an absolute **2.6e-17**.

The fixture is the only reason this was noticed rather than shipped. A block that is
exactly another block in different units is a strange thing to generate deliberately, and
it is the only test that can fail this way.

### G8.4 — the affinity diagonal made `graph` produce nothing, twice · **blocks** · fixed

The first `graph` output separated *neither* planted partition: 1.005 and 1.004, against
1.39 and 1.23 after the fix. The fused affinity matrix was structurally correct — its
within-partition mean was 2.5x its between-partition mean — and the profile built from it
carried no structure at all.

Equation 8 pins every protein's self-affinity at exactly 1/2 while its affinities to
everything else are O(1/N); here, 0.5 against 0.001. Reading rows as feature vectors, that
single entry sits in a different column for every protein, so the euclidean distance
between any two rows is about sqrt(2)/2 regardless of what the rest of the row says.
Every pair equidistant, every partition invisible.

The same defect had the same effect on the per-protein shares, which were computed as an
inner product between the fused matrix and each block's initial affinity: the shared 0.25
diagonal term dominated both, and an informative block against a pure noise block came out
**0.500 / 0.499**. Off-diagonal, it is **0.597 / 0.403**.

Both are fixed by dropping the diagonal, which discards no information — the self-affinity
is a normalization constant, identical for every protein, and says nothing about who
anyone is near. What is worth recording is that neither failure would have been visible
without a fixture carrying a known partition. Every shape was right, every value was
finite, the matrix was symmetric and row-stochastic, and the answer was noise.

### G8.5 — FOLLOWUPS #16 was about the wrong end of the pipe · **ADR corrected**

ADR 0002 names per-protein weighting as `graph`'s unique advantage. `SpaceSpec.weights` is
one scalar per block, so #16 recorded the ADR as overclaiming and Gate B deferred it here.

The deferral was right and the framing was not. Per-protein weights are not a config
input, and could not usefully be one: it would ask the user to know, per protein, which
kind of evidence deserves trust — which is the output of the analysis, not an input to it.
They are something SNF *produces*. The fused network gives every protein its own mixture
of the blocks, recovered afterwards as `BlockContribution.per_protein_share`.

Demonstrated rather than asserted: an informative block against a noise block, both
weighted 1.0. `late` reports 50/50 because that is what the weights say. `graph` reports
59.7/40.3, and does so for every protein individually — the informative block wins in
every one of the 240 rows, with a per-protein range of 0.565 to 0.617.

ADR 0002's second cost — "one optional dependency (`snfpy`)" — is also wrong, and for a
reason the ADR could not have known: ADR 0006 would make `graph` the one strategy that
silently disappears in the bare environment. It is 60 lines of numpy instead. The real
cost is that there is no reference implementation here to check the output against, so the
tests check the algorithm's properties. Stated in ADR 0013 §4 rather than left implicit.

### G8.6 — the entry point crashed on the demo. Fourth sighting

19 unit tests passed and `snakemake` died at rule 26. `coregister.py` imports
`features_for` from `reduce_space`, whose return signature had gained a third element —
and separately read `.features` off what it assumed was a single `BlockResult`, which was
correct only while spaces were single-block.

Both are the same lesson as G7.8, G6.4 and `compute_block` before them: **a test suite that
passes is not evidence the entry point works.** The rule is now in CLAUDE.md and it has
now caught four defects. The fix also improved the thing it broke — a co-registered fused
space compares its fused geometry rather than one of its blocks, which is what it should
always have done — and turned up a third defect nobody was looking for: `coregister`'s
manifest named one block per space as an input, so a changed second block would have
looked like a cache hit.

`tests/test_reduce_space.py` is new and did not exist before this group. `reduce_space.py`
had no direct tests at all; the demo was its only exercise.

### G8.7 — the demo's two blocks are in different orders, so the alignment is load-bearing

While diagnosing G8.4 I fused the demo's `tmscore` and `biophys` blocks directly, without
going through `features_for`, and got a different convergence profile. The cause is that
the two blocks list the same eleven proteins in different orders — `tmscore` starts at
`A0A286Q506`, `biophys` at `Q6QAQ1` — because one comes from the similarity matrix and the
other from `uniprot_features.tsv`.

So the intersection-and-align step inside `reduce_space` is not defensive padding on this
cohort; without it every fused distance would pair a protein with a different protein, and
the result would still be a well-formed square matrix of plausible numbers. This is ADR
0007's rule met in a third place, and `test_a_reordered_block_is_realigned_not_read_
positionally` is the test that fails when it is removed.

### G8.8 — a params validator, and what it caught in the first second

`SpaceConfig.params` was a free-form mapping, so a misspelled `iteratons` would reach the
manifest looking configured while the run used the default. `STRATEGY_PARAMS` now declares
what each strategy consumes and the validator rejects the rest at parse time.

It failed the suite immediately: `test_config_schema.py`'s full example config has carried
`params: {K: 20}` on a `graph` space since the schema was written — uppercase, matching
the paper's notation, and read by nothing because nothing read `params` at all. Fifth
sighting of the pattern behind #29, #32 and G6.4, and the first one caught structurally
rather than by remembering to look.

### G8.9 — what fusing the demo actually found

Worth reading, and it is not a biological result — the cohort is eleven actins.

`early` gives `tmscore` **73.3%** of the variance against `biophys`'s 26.7%. That is 11
columns against 4 and nothing else, and it is over the 70% threshold, so both warnings
fire. (`early` had no dominance warning until this run; only `late` and `graph` did. 73.3%
is exactly the case where one warning without the other is easy to skim past.)

`graph` needed `iterations: 200`. The paper recommends 10-20, and at N=11 with k=5 the
last step still moves an affinity by 0.029 after 20 — then 5.4e-07 after 100 and 2.8e-17
after 500. Convergence is monotone in both `k` and `iterations`, and small `k` on a small
cohort is the slow corner. The warning is what surfaced it; the demo config now sets a
value that converges, because a demo should not ship a result its own output calls
provisional.

The interesting one is the co-registration. `fused_late` sits **between its two parents**:
it has the highest neighborhood Jaccard of any pair with `structure` (0.789, median 1.0)
and the highest rank correlation of any pair with `physicochemistry` (0.808). So the two
"which block dominates" measures point in opposite directions — the realized share says
`biophys` at 66%, the neighborhoods say `structure`. Both are true and they answer
different questions: the share is about the magnitude of the distances, the Jaccard is
about their ordering. A reader who has only one of the two numbers will over-read it,
which is the argument for printing both.

### Verification, group 8 first half

- Unit suite: 815 passed, 64 skipped (bare env). The 5 new skips are scipy- and
  sklearn-gated; both files were run where they do not skip — `test_fusion.py` at
  **59 passed / 0 skipped** and `test_reduce_space.py` at **22 passed / 0 skipped**, in
  `.snakemake/conda/21cb44d…`, which carries pytest, scipy 1.13.1, scikit-learn 1.2.2 and
  umap 0.5.3. That env is the full stack, not only scipy — worth knowing, and G7.12
  understated it.
- **Parity: 33 slow tests, 0 differing files**, against `upstream/main` at `36a38c7`.
  Expected and checked rather than assumed: the default configuration defines no spaces,
  so `reduce_space` does not execute in a parity run at all.
- Lint: `ruff check`, `ruff format --check`, `snakefmt --check` all clean.
- Cluster DAG 16 and search DAG 25, both unchanged. Multispace demo 25 → 28 — one rule per
  fused space — and it runs **28/28** to completion.
- **The Snakefile needed no change.** The `reduce_space` rule already took every block of
  its space as input, so fusion is a config-reachable capability rather than new
  machinery. That is worth noting as evidence the group 5 wiring was general enough.
- Each of the four commits verified alone in a detached worktree: **735 / 795 / 815 / 815**
  passed, lint clean, cluster DAG 16 at every point.
- **Mutation harness exits 0** — 17 mutations across three scales, 12 detected, 5
  survived as expected with the reason recorded against each, **0 unexplained holes**.
  Unchanged from group 7b, and the tree was checked restored afterwards: no diff against
  HEAD in any source file.
- No new mutation entries, for the reason G7.6 records: the default configuration defines
  no spaces, so the parity test cannot see any of this output and every mutation to it
  would survive for a structural reason rather than a real hole. Checked rather than
  assumed — the parity run below is the same 0 differing files it was before fusion
  existed.
- Pre-existing files touched stays at **10**. This half edits no pre-existing file at all
  — `reduce_space.py`, `coregister.py`, `config_schema.py` and `demo/multispace/config.yml`
  are all files this work created, and the Snakefile was not touched.
- Still open for group 8's second half: the nine diagnostics, the four
  `DiagnosticsConfig` fields that are read by nothing (FOLLOWUPS #36), and the clustering
  decision that blocks cross-space ARI.

## Group 8, second half — diagnostics

Phase 5's clustering-free diagnostics: block redundancy (item 1), trustworthiness and
continuity (item 4), self-diff determinism (item 9), plus the wiring of censoring
(item 5) and cohort reporting (item 6). ADR 0014 records what it decided. Items 3, 7
and 8 are deferred to group 8c with the clustering decision they depend on.

### G8b.1 — Item 6 was already built, and the plan said it was not

PLAN §0.5's table listed Phase 5 item 6, cohort diagnostics, as "not built". It is
built. `cohort.CohortReport` records candidates before filtering, candidates before and
after truncation, the selection rule, whether truncation was reproducible, and the
taxonomic composition of retained against discarded proteins; `download_pdbs` declares
`cohort_report.json` as a rule output and writes it. That is item 6's full text.

Checked rather than assumed, which is the only reason it was found — the alternative
was a second implementation of a report that already existed. Group 8b copies the file
into the space report instead, and only in search mode, because cluster mode makes no
cohort decision and the absence there is correct rather than a gap.

Three of Phase 5's nine were therefore already done before this group started: item 2
(group 8a), item 5 (group 4, unwired), item 6 (group 3, wired).

### G8b.2 — The fixture caught two overclaims in its own commit

`tests/embedding_cohort.py` plants a 2x2 table: one truth, four maps of it, and each of
the four cells (trustworthy?, continuous?) occupied. The off-diagonal cells are the
point. Trustworthiness and continuity are near-mirror formulas over the same two
neighbor sets, so the likeliest defect is computing one of them twice under two names
or swapping the labels — and neither is visible in a fixture where the map is simply
good or simply bad, because those give `T == C` and a wrong answer that agrees with
itself reads as two agreeing answers. This is G8.4's lesson from the other side.

Writing the fixture's own tests, before any statistic existed, falsified two claims the
fixture's docstring made:

- **"Within-half distances are preserved exactly."** They are not, as *measured*.
  `split` translates a random half by 500 units, and `(a + 500) - (b + 500)` cancels
  inexactly for a and b of order 1 — measured residual 8.1e-13 relative. The
  *translation* is bitwise exact; the recomputed distance is not. The exact claim now
  sits on the translation, where it is true, with a separate test at the tolerance the
  recomputation actually needs.
- **An integer-lattice truth.** The first draft laid the points on a 1-D lattice, where
  every point is exactly equidistant from its two neighbors. Exact ties make the
  k-nearest set depend on how the sort breaks them: the isometric case scored 0.9993
  instead of 1.0 and disagreed with scikit-learn by 2.6e-3. That is a sorting artifact
  and is indistinguishable from a formula error. Drawn from a continuous distribution
  there are no ties, the isometric case is exactly 1.0, and agreement with scikit-learn
  is exact.

Third group running for a fixture built before the thing it measures, and the third
time it paid before that thing existed (G7.7, G8.4).

### G8b.3 — The tolerance question, answered by measurement rather than by habit

CLAUDE.md's rule is that an absolute tolerance of 1e-12 is vacuous on rows where the
values are tiny. Both cross-checks here hit the inverse of that problem and it is worth
recording, because the reflex fix would have been wrong in both.

Trustworthiness lives in `[0, 1]`, so `abs(mine - theirs) < 1e-9` would pass on an
answer that was wrong by a factor the ties artifact produces. Exact equality was
available in the prototype and is *not* available in the shipped code, because
computing per-protein values and averaging associates the sum differently from
scikit-learn's single global sum. Measured across four cases at k = 5, 10 and 20: exact
in 7 of 12, one unit in the last place in the other 5, worst relative difference
2.2e-16 — one machine epsilon. The asserted tolerance is 1e-13 relative: three orders
above the observed noise, thirteen orders below the 2.6e-3 that a real defect of the
kind this fixture was built to avoid produces. Where equality *is* available — the
isometric case, both sides exactly 1.0 — it is asserted separately rather than hidden
under the tolerance the other cases need.

Redundancy has the opposite shape. A correlation of -0.0039 is an ordinary value here,
produced by two independent blocks, and an absolute tolerance would pass on an
implementation that returned zero for it. Relative again, at 1e-12, against scipy's
`pearsonr` and `spearmanr`.

### G8b.4 — `fusion_cohort` was already the right fixture for redundancy

Group 8a built it for something else and it needed no extension. It plants two exactly
crossed partitions, so `wide` and `narrow` are independent by construction and must
correlate at zero — measured -0.009 Pearson, -0.004 Spearman. And it carries
`narrow_rescaled`, which is `narrow` times 1000, so two of its blocks are the same
information in different units and must correlate at exactly one — measured Spearman
exactly 1.0, Pearson within 1.1e-16.

Those are the two ends of the scale the diagnostic measures, both planted, and group 8a
arranged neither of them for this purpose. Worth recording as an argument for the
fixture-first discipline that is not "it caught a bug": a fixture built to pin one
thing's right answer is frequently the only honest test data available for the next
thing, and a fixture built to pin shapes never is.

### G8b.5 — Two defects the unit tests found, one the entry point found

**Found by a unit test, would have crashed a rule.** `_pearson` returned `np.float64`
rather than `float`, because `np.sqrt` returns a numpy scalar and dividing a Python
float by one gives a numpy scalar back. `np.float64 >= threshold` is a `np.bool_`, and
`json.dump` refuses it. The redundancy report goes into a manifest, so this would have
surfaced as a crashed rule rather than as a wrong number — but it would have surfaced
in the demo, not here, if the test had not serialized the report.

**Found by a unit test, and it was the test that was wrong.** The
translation-invariance test shifted `narrow` — a block whose values are of order 0.01 —
by a flat 17.5, and asserted `rtol=1e-12`. It failed at 2.6e-10. The invariance is
exact; the failure was four digits of cancellation in the distance computation, from a
shift/scale ratio of about 6000. The shift is now a multiple of each block's own scale,
which puts the test back on the property it names.

**Found by the demo, and nothing else could have found it.** Fifth sighting of "a
passing test suite is not evidence the entry point works". Every unit test ran at
N=240. The demo cohort is eleven proteins and `DEFAULT_K` is fifteen, and the statistic
is undefined for `k >= (2N-1)/3` — so all seven spaces failed at once, in the first
end-to-end run. `k` is now clamped to the cohort with the request kept and reported
(ADR 0014 §6), which is the idiom `reduce_pca` already uses for `n_components`. Four
tests now run at N=11 and N=12; the sixth sighting will not be this.

### G8b.6 — `from_legacy` was dropping the entire `diagnostics:` key

Found by the test PLAN's FOLLOWUPS #36 rule demanded, within minutes of writing it —
which is the same way ADR 0013 §6's `STRATEGY_PARAMS` validator found a dead
`params: {K: 20}` (G8.8).

`from_legacy`'s legacy branch — the one a plain cluster-mode config takes — constructs
its `MultispaceConfig` from a literal dict and carried `cohort` and `enrichment`
through it but not `diagnostics`. So a legacy config could set `diagnostics.k` and be
silently ignored, or misspell a diagnostics key and never be told, because the key
never reached `_reject_unknown_keys`.

The part worth keeping: **three lines above the omission is a comment explaining that
`enrichment` is carried through here precisely so this does not happen to it**, naming
`_reject_unknown_keys` and calling it "the exact failure" to prevent. The comment was
right, was read by whoever wrote it, and the next key added below it was dropped
anyway. A comment that states an invariant is not an enforcement of it — which is the
whole argument for `STRATEGY_PARAMS`-style checks, now made twice.

### G8b.7 — What the demo actually reports, and it is a verdict against the demo

The multispace demo runs 35/35 and every map in it scores near chance: trustworthiness
0.34 to 0.76, continuity 0.24 to 0.78, and between 4 and 10 of the 11 proteins flagged
as having positions that should not be read.

That is the correct answer, not a defect, and it is the first time this work has
produced a number that criticizes its own demo. Eleven proteins cannot support a 2-D
embedding: `k` clamps from 15 to 6, which is more than half the cohort, so
"neighborhood" has stopped meaning anything local. Every picture the demo draws is a
picture of eleven points and these are the numbers that say so. Recorded in
`demo/multispace/config.yml` rather than left in a log.

Two more measured, both recorded there:

- `tmscore` and `biophys` correlate at Spearman **0.883** over their pairwise
  distances — just under the 0.90 threshold. The three fused spaces report an honest
  50/50 contribution split, and the pair of numbers is the point: the split is correct
  arithmetic about two blocks that are closer to saying the same thing than their names
  suggest. This is exactly the gap ADR 0002's `early` warning gestures at and cannot
  itself fill.
- The censoring rate on the demo matrix is exactly **0.000**. At N=11 Foldseek's
  per-query cap of 1000 cannot bind, so there is no fill. Independent confirmation of
  the mutation-testing finding that the demo fixture cannot exercise the censoring path
  (§0.1), arriving from the other direction.

### G8b.8 — Exactness that does not survive the store

`narrow` against `narrow_rescaled` correlates at exactly 1.0 in the unit test and at
0.9999999999949 through the entry point. The block store writes float32 (ADR 0004), and
quantizing two copies of the same data at scales 0.01 and 10 is not a proportional
operation: measured deviation 9.7e-8 relative, one float32 epsilon, which is enough to
swap a handful of the 28,680 distance ranks.

Not a defect in either place. Recorded because the two tests assert different things
about the same pair of blocks and the entry-point one asserts *less*, which looks like
carelessness unless the reason is written down. The exactness is a property of the
arrays; asserting it after a round trip through the store would be asserting that the
store does something ADR 0004 says it deliberately does not.

### G8b.9 — The determinism guard carries its own negative control

A determinism check is the easiest kind of test to write in a form that cannot fail,
because the thing it looks for is usually absent. scikit-learn picks its SVD solver from
the input shape, and only the randomized solver it selects above 500 samples is
nondeterministic without a seed — so the identical guard on a 400-protein fixture
passes unconditionally and forever.

`test_the_guard_can_actually_fail_at_this_fixture_size` configures PCA the way this
repository deliberately does not, `auto` and unseeded, and requires the two runs to
disagree. They do, at N=750, which independently reconfirms the measurement the fixture
size was chosen from. Its failure message says what its own passing would mean.

This is the general form of the rule PLAN states as "a diagnostic that can never fire is
decoration": for a *guard*, the corresponding check is a test that the guard's subject
can actually occur.

### Verification

- Unit suite: **949 passed, 89 skipped** (bare env, no sklearn/scipy/umap).
- Gated tests where they do not skip, in `.snakemake/conda/21cb44d…`:
  `test_diagnostics_embedding.py` + `test_diagnostics_redundancy.py` **92 passed,
  0 skipped**; `test_determinism.py` **8 passed, 0 skipped** with `--runslow`, 42s.
- Lint: `ruff check`, `ruff format --check`, `snakefmt --check` all clean.
- DAGs: cluster **16**, search **25**, both unchanged. Multispace **28 → 35**, one rule
  per space, and it runs **35/35** end to end.
- Pre-existing files touched: still **10**, recounted rather than assumed. This group
  adds none to that count: it edits `Snakefile`, which group 5 already put in the diff,
  and no other pre-existing file. Everything else it touches is a file this work
  created.

## Gate D — after commit group 8 (fusion and diagnostics)

*Not yet run.*

## Gate E — before opening the PR

*Not yet run.*
