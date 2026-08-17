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

*Not yet run.*

## Gate C — after commit group 5 (the port)

*Not yet run.*

## Gate D — after commit group 8 (fusion and diagnostics)

*Not yet run.*

## Gate E — before opening the PR

*Not yet run.*
