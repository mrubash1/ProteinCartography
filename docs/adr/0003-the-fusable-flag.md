# ADR 0003 — The `fusable` flag, and why it must not be helpfully removed

Status: accepted
Date: 2026-08-16

## Context

Once blocks can be fused into a geometry (ADR 0002), *any* per-protein quantity
becomes a candidate axis. Some of them must never be one, and the reasons are not
obvious from the code — which means the next maintainer, seeing a flag that
prevents a thing from working, will be tempted to remove it.

This ADR exists to make that argument un-removable by writing it down.

**Three distinct reasons a signal must stay overlay-only:**

**1. Circularity.** Fusing taxonomy into a geometry makes every taxon-specific
cluster claim circular: the clusters separate by taxon because taxon was an axis.
The same applies to patristic distance, and to any residual derived from the
geometry itself.

**2. Artifact promotion.** Fusing pLDDT, censoring rate, or disorder fraction
makes *protein length* a principal axis of a biology map. All three correlate
strongly with length, and none of them is a statement about the protein's
biology. A map whose first axis is "how long is it and how confidently was it
predicted" looks like a finding and is not one.

**3. Measurement-of-the-measurement.** Neighborhood stability and censoring rate
are properties of *how well we measured this protein*, not of the protein. Fusing
a QC metric into the thing it is qualifying is a category error.

## Decision

**Every `BlockSpec` carries `fusable: bool` and, when false,
`not_fusable_reason: str`.**

**The config validator rejects a `fusable: false` block in any multi-block space,
with an error that states the reason** rather than a generic type complaint:

```
Block 'taxonomy' cannot be fused into space 'multiview'.
Reason: fusing taxonomy makes every taxon-specific cluster claim circular —
the clusters would separate by taxon because taxon was an axis.
Taxonomy is available as an overlay on any space.
```

Enforcement is at the schema layer, not at the fusion layer, so it fails at
config-load time rather than after an expensive computation.

**The initial `fusable: false` list:**

| Block | Reason |
|---|---|
| `taxonomy` | circularity — every clade-enrichment claim becomes circular |
| `phylogeny` / patristic distance | circularity, same argument |
| `pLDDT` / `pdb_confidence` | artifact promotion — strongly length-correlated |
| `censoring rate` | measurement-of-the-measurement, and length-correlated |
| `disorder` | artifact promotion — explains most TM-score failures, so fusing it makes length an axis |
| neighborhood stability | measurement-of-the-measurement |
| any geometry-derived residual | circularity by construction |
| **`struclusters`** | **see below** |

**`struclusters` is added to this list as a result of the Phase 0.5 claim B
finding.** The evidence, verified twice against the pinned Foldseek binary:

- `foldseek clust` is handed `foldseek_out`, the **search alignment** database.
  It is never handed `foldseek_tmscore`. The two branches diverge and never
  rejoin.
- It could not be handed the TM-score DB even deliberately: dbtype `0x66` is
  rejected by `clust`, which accepts only `0x05` Alignment and three others.
  **Clustering on TM-score is not expressible with this module.**
- `--similarity-type 2` is documented by the pinned binary (foldseek
  `6.29e2557`) as *"sequence identity"*, and the stored identity was measured to
  be **amino-acid** identity (matching in 121/121 demo alignments, versus 15/121
  for 3Di).

**The accurate characterization — use this wording, not "sequence-derived":**

> `struclusters` is a **structurally-gated graph with sequence-weighted
> clustering.** Membership candidates come from a 3Di+AA structural alignment on
> a 3Di k-mer prefilter, so two proteins with high sequence identity that never
> align structurally never become an edge. Amino-acid identity only *weights*
> those edges inside the greedy set-cover.

It is therefore not a pure sequence label, and not the structural label its name
implies. **Consequences:**

1. `fusable: false`, because contrasting "structure space" against "sequence
   space" using `struclusters` as structural ground truth is partly circular.
2. Every co-registration comparison that uses `struclusters` as a structural
   reference is audited (Phase 6).
3. The block-redundancy diagnostic explicitly checks `struclusters` against
   `tmscore` before any result leans on the distinction (Phase 5).
4. The naming problem is upstream's and is logged separately in
   `docs/FOLLOWUPS.md` #9 as a candidate standalone PR. **The sharpest statement
   of it:** the pipeline computes TM-scores, uses them for the embedding
   *coordinates*, then *colors* those points with an amino-acid-identity-weighted
   label, in a rule named `plot_similarity_strucluster`. The axes and the colors
   come from different similarity measures.

## Consequences

- Some things users will want to fuse, they cannot. The error explains why and
  points at the overlay path, which is always available.
- The flag is enforced in one place, so there is exactly one code path to audit
  in Gate D ("can a `fusable: false` block reach a geometry by any route?").
- A block author must make a deliberate choice. The default is `fusable: true`,
  because most representations are legitimately fusable and a false default
  would be noise.
- **This ADR is the artifact that prevents the flag being removed as an obstacle.**
  A future maintainer who finds it inconvenient will find the reasoning here
  rather than having to reconstruct it from a validator error.

## Alternatives rejected

**Warn instead of reject.** Rejected: warnings in a pipeline that prints
thousands of lines are not read, and the failure mode is a *confidently wrong
scientific claim*, not a degraded plot.

**Document the guidance in the README and let users decide.** Rejected on the
rule this work applies throughout: an invariant that is only documented is an
invariant that is violated. The circularity is
non-obvious enough that a well-intentioned user will get it wrong.

**Allow fusion with an explicit `i_know_what_im_doing: true` escape hatch.**
Rejected for now. There is no demonstrated legitimate use, and the hatch would
appear in copied configs and spread. It can be added later if a real case
appears; it cannot easily be removed once it exists.

**Leave `struclusters` fusable, since it is structurally gated after all.**
Rejected. The gating makes it *not purely sequence*, which is a correction to how
we describe it — it does not make it safe as structural ground truth in a
structure-versus-sequence contrast, which is the specific use this project puts
it to.
