#!/usr/bin/env python
"""What each panel is plotting, in words, for the person reading the page.

Every panel on the explorer gets a fold-out that answers three questions: what
is on the axes, what the numbers are, and what the picture cannot be read for.
The strings live here rather than in the template so they can be tested, and so
that each one can be checked against the code that produces the number.

**These strings ship to biologists, so a plausible-but-wrong sentence is worse
than no sentence at all.** Two rules follow, and they are the only interesting
thing about this module:

* Anything the code does not settle is written as ``NOT_DETERMINABLE`` --
  literally the words "not determinable from the code" -- and never guessed. A
  reader can act on "the code does not say"; they cannot act on a confident
  sentence that turns out to be someone's inference.
* **Any number that varies between cohorts is passed in, never typed here.**
  The fused map's realized shares are 0.4405/0.5595 for the 367-protein
  chymotrypsin cohort and 0.4208/0.5792 for the 308-protein actin one. A
  constant in this file would therefore be wrong on one of the two panels it
  was rendered into, and wrong in a way nobody would notice, because it would
  still look like a measurement. The block manifests already carry these facts
  per cohort, so `describe_space` takes them as arguments.

Deliberately dependency-free, like `panels.py`: `test_optional_dependencies`
asserts the explorer imports in an environment with nothing optional installed,
and a description module that needed pandas to say what an axis means would be
the silliest possible way to break that.
"""

from __future__ import annotations

__all__ = [
    "NOT_DETERMINABLE",
    "AXES",
    "describe_space",
    "describe_panel",
]

#: Written wherever the code does not settle the question. Never paraphrased --
#: a test greps for this exact string, so a future edit that replaces it with a
#: guess has to delete the phrase to do it.
NOT_DETERMINABLE = "not determinable from the code"


def _description(paragraphs=(), hazards=(), sources=()) -> dict:
    """One fold-out, in the single shape the template knows how to render.

    Three lists rather than one blob so the template can style a hazard
    differently from an explanation. A hazard rendered in body text is a hazard
    a reader skims past, which is the failure mode this whole module is about.
    """
    return {
        "paragraphs": [p for p in paragraphs if p],
        "hazards": [h for h in hazards if h],
        "sources": [s for s in sources if s],
    }


# ---------------------------------------------------------------------------
# the axes, which are the same on every map this pipeline draws
# ---------------------------------------------------------------------------

#: Appended to every space description. The single most common misreading of a
#: UMAP is treating the gap between two clusters as a quantity, and it costs one
#: sentence to say that it is not.
AXES = _description(
    paragraphs=[
        "Both axes are UMAP1 and UMAP2 of a 30-component PCA of this block's "
        "features -- `pca_umap` means PCA-30 and then UMAP. They have no units "
        "and neither axis means anything on its own: rotating or reflecting the "
        "picture would be an equally correct drawing of the same result.",
        "Nearby points are the readable part, and only as far as the "
        "faithfulness diagnostics in this panel's banner allow. The distance "
        "between two well-separated clusters is not interpretable, and neither "
        "is the direction from one to the other.",
    ],
    sources=["reduce_space.py:46", "spaces/reducers/core.py:106"],
)


# ---------------------------------------------------------------------------
# the blocks, keyed by provider
# ---------------------------------------------------------------------------


def _biophys(facts: dict, params: dict) -> dict:
    descriptors = list(facts.get("descriptors") or params.get("descriptors") or ())
    ph = facts.get("ph", params.get("ph", 7.0))
    column = params.get("sequence_column", "Sequence")
    return _description(
        paragraphs=[
            "Four physicochemical scalars per protein, computed from the "
            f"`{column}` column of the cohort's feature table"
            + (f" ({', '.join(descriptors)})." if descriptors else "."),
            "`gravy` is mean Kyte-Doolittle hydropathy, roughly -4.5 to +4.5, "
            "high meaning hydrophobic. `aromaticity` is the fraction of "
            "residues that are F, W or Y, so 0 to 1. `isoelectric_point` is the "
            "pH of zero net charge, found by bisection over a Bjellqvist-style "
            "pKa table, in pH units. `charge_per_residue` is net charge at pH "
            f"{ph} divided by the number of standard residues, roughly -0.1 to "
            "+0.1.",
            "Two proteins' distance is the plain Euclidean distance between " "those four numbers.",
        ],
        hazards=[
            'The block declares `normalization="zscore_within"` and nothing '
            "reads `spec.normalization` (FOLLOWUPS #32), so the four columns "
            "enter the distance on their own scales. Isoelectric point spans "
            "about 4 to 12 pH units while charge per residue spans about 0.2, "
            "so the Euclidean distance is dominated by isoelectric point and "
            "this map is close to a map of pI. Read it as one.",
        ],
        sources=[
            "blocks/biophys.py:276",
            "blocks/biophys.py:312",
            "blocks/biophys.py:527",
        ],
    )


def _threedi(facts: dict, params: dict) -> dict:
    k = facts.get("k", params.get("k", 3))
    scaling = facts.get("scaling", params.get("scaling", "frequency"))
    n_kmers = facts.get("n_kmers")
    width = (
        f"{n_kmers} columns for this cohort"
        if isinstance(n_kmers, int)
        else "one column per observed k-mer"
    )
    return _description(
        paragraphs=[
            "Foldseek's 3Di structural alphabet, read as text. Each protein "
            f"becomes the profile of its 3Di {k}-mers, with `scaling="
            f'"{scaling}"`, so a row is relative frequencies rather than counts '
            "and a long protein is not automatically far from a short one.",
            "Only k-mers observed somewhere in this cohort become columns, "
            f"which is {width}. The width is therefore a property of the "
            "cohort and not a constant: the same provider gives 4982 columns "
            "for the 367-protein chymotrypsin cohort and 4594 for the "
            "308-protein actin one. Two cohorts' 3Di spaces are not embedded "
            "in the same feature space and their coordinates cannot be "
            "compared.",
        ],
        sources=["blocks/threedi.py:69", "blocks/threedi.py:152"],
    )


def _domains(facts: dict, params: dict) -> dict:
    source = params.get("source", "pfam")
    families = facts.get("n_families")
    if not isinstance(families, int):
        listed = facts.get("families")
        families = len(listed) if isinstance(listed, list) else None
    unannotated = facts.get("proteins_without_domains")
    if not isinstance(unannotated, int) and isinstance(unannotated, list):
        unannotated = len(unannotated)
    counted = (
        f"This cohort has {families} distinct {source.title()} families as columns"
        if isinstance(families, int)
        else f"One column per {source.title()} family seen in this cohort"
    )
    if isinstance(unannotated, int):
        counted += f", and {unannotated} protein(s) carry no annotation at all."
    else:
        counted += "."
    return _description(
        paragraphs=[
            f"Binary {source.title()} domain presence: one column per family, 1 "
            "if this protein is annotated with it and 0 otherwise. " + counted,
            "Distance is Euclidean on those 0/1 vectors, which is the square "
            "root of the number of families two proteins do not share. "
            "`jaccard` -- the right distance for sets -- is refused with an "
            "error rather than silently approximated, because the reducer core "
            "is not metric-aware.",
        ],
        hazards=[
            "An all-zero row does not mean 'this protein has no domains'. It "
            "means nobody has annotated one. Absence of annotation and absence "
            "of domain are the same vector here, and they sit together in the "
            "same corner of the map.",
            "Tie degeneracy is why this space is often unreadable. Identical "
            "presence vectors collapse onto the same point before the reducer "
            "runs, and where UMAP puts a pile of identical rows says nothing "
            "about those proteins. Measured on the 308-protein actin cohort, "
            "187 proteins carry exactly PF00022 and nothing else, leaving 65 "
            "distinct rows for 308 proteins.",
        ],
        sources=["blocks/domains.py:165", "blocks/domains.py:204"],
    )


def _tmscore(facts: dict, params: dict) -> dict:
    representation = params.get("representation", "profile")
    if representation == "profile":
        body = [
            "`representation: profile`. A protein has no feature vector of its "
            "own -- its feature vector is its **whole row** of the all-vs-all "
            "similarity matrix. Two proteins land near each other when they are "
            "similar to the same other proteins, which is a second-order "
            "similarity and not a direct structural distance.",
            "Distance is therefore Euclidean between two similarity profiles. "
            "It is not the pairwise score, and it is not on the score's scale.",
        ]
    else:
        body = [
            "`representation: direct`. The similarity matrix is read as a "
            "distance matrix through `1 - TM`, one number per pair.",
        ]
    return _description(
        paragraphs=body,
        hazards=[
            "The scores in the matrix come from foldseek's 3Di+AA alignment, "
            "not from TM-align (FOLLOWUPS #60). 'TM-score' is the column name "
            "the pipeline inherited; it overstates what was computed.",
        ],
        sources=["blocks/tmscore.py:47", "blocks/tmscore.py:172"],
    )


#: provider name -> a function of (manifest extra, block params).
#:
#: Keyed on the *provider* and not the block id, for the reason
#: `config_schema.NOT_FUSABLE_PROVIDERS` gives: the block id is a free-text key
#: the user chooses, so a table keyed on it describes `tmscore:` correctly and
#: says nothing at all about `tm:`.
BLOCK_DESCRIBERS = {
    "biophys": _biophys,
    "threedi": _threedi,
    "domains": _domains,
    "tmscore": _tmscore,
}


def describe_block(provider: str, facts: dict = None, params: dict = None) -> dict:
    """One block's fold-out, or an honest blank for a provider with no entry."""
    describer = BLOCK_DESCRIBERS.get(provider)
    if describer is None:
        return _description(
            paragraphs=[
                f"This block comes from the `{provider}` provider. What it puts "
                f"on the axes is {NOT_DETERMINABLE} here: no description has "
                "been written for that provider, and guessing one would be "
                "worse than saying so."
            ]
        )
    return describer(dict(facts or {}), dict(params or {}))


# ---------------------------------------------------------------------------
# the fusion strategies
# ---------------------------------------------------------------------------


def _share_sentence(contributions) -> str:
    """The realized shares of THIS cohort's fused map, from its own manifest.

    Returns "" when the shares are absent rather than inventing them. See the
    module docstring: the two cohorts on the shipped page realize different
    shares from the same nominal 50/50, so this sentence cannot be a constant.
    """
    parts = []
    for entry in contributions or ():
        realized = entry.get("realized_share")
        asked = entry.get("share")
        if realized is None:
            continue
        text = f"{entry.get('block_id')} {100 * float(realized):.1f} %"
        if asked is not None:
            text += f" against a requested {100 * float(asked):.0f} %"
        parts.append(text)
    if not parts:
        return ""
    return "For this cohort the realized shares are " + "; ".join(parts) + "."


def describe_late(contributions=()) -> dict:
    return _description(
        paragraphs=[
            "Late fusion: D^2 = sum_i w_i * d~_i^2 / sum_i w_i, computed after "
            "each block's distance matrix has been divided by its own mean "
            "off-diagonal distance. Feature-space structure is gone by "
            "construction -- you can read off which **block** drove a distance, "
            "never which column.",
            "An equal weight is a request, not an outcome. mean(d~^2) is "
            "1 + var(d~), so a block whose normalized distances are widely "
            "dispersed contributes more to the fused squared distance than its "
            "weight asked for, and a block whose pairs are all equally far "
            "apart contributes exactly its weight and no shape. " + _share_sentence(contributions),
        ],
        hazards=[
            "Fusion is a named analysis, never the default view (source "
            "document 1.06). A fused map answers 'which proteins are alike "
            "across these blocks at once', and it cannot answer the question "
            "co-registration exists for -- where two blocks disagree -- because "
            "the disagreement has been averaged away.",
        ],
        sources=["fusion.py:565", "fusion.py:551"],
    )


def describe_strategy(strategy: str, contributions=()) -> dict:
    """The fold-out for how a space combined its blocks, if it combined any."""
    if strategy in (None, "", "none"):
        return _description()
    if strategy == "late":
        return describe_late(contributions)
    return _description(
        paragraphs=[
            f"This space fuses its blocks with `strategy: {strategy}`. How that "
            f"strategy weighs the blocks is {NOT_DETERMINABLE} here -- no "
            "description has been written for it. The contribution line above "
            "the map is measured and is the thing to read."
        ]
    )


# ---------------------------------------------------------------------------
# the assembled space description
# ---------------------------------------------------------------------------


def describe_space(space_id: str, strategy: str = "none", blocks=(), contributions=()) -> dict:
    """The fold-out for one map on the grid.

    Args:
        space_id: the space's own name, used only in the lead sentence.
        strategy: the fusion strategy, as the config resolved it.
        blocks: in order, ``{"block_id", "provider", "params", "facts"}`` per
            block. ``facts`` is the block manifest's ``extra``, which is where
            the cohort-dependent numbers -- the 3Di vocabulary size, the Pfam
            family count -- actually live.
        contributions: the space's measured per-block shares, for a fused space.
    """
    paragraphs, hazards, sources = [], [], []
    named = [b.get("block_id") or b.get("provider") for b in blocks]
    if named:
        paragraphs.append(f"`{space_id}` is built from " + ", ".join(f"`{n}`" for n in named) + ".")
    for block in blocks:
        part = describe_block(block.get("provider"), block.get("facts"), block.get("params"))
        paragraphs.extend(part["paragraphs"])
        hazards.extend(part["hazards"])
        sources.extend(part["sources"])
    fused = describe_strategy(strategy, contributions)
    paragraphs.extend(fused["paragraphs"])
    hazards.extend(fused["hazards"])
    sources.extend(fused["sources"])
    paragraphs.extend(AXES["paragraphs"])
    sources.extend(AXES["sources"])
    # Deduplicated in place, because a fused space names two blocks and the
    # same file would otherwise be cited twice in one footer.
    return _description(paragraphs, hazards, list(dict.fromkeys(sources)))


# ---------------------------------------------------------------------------
# the catalogue panels
# ---------------------------------------------------------------------------

#: What the source document itself says each panel is for, quoted rather than
#: paraphrased, plus what is different about this pipeline's version of it.
#: Quoted because the panel's whole claim to exist is that the source argued for
#: it; a paraphrase would let the panel drift away from the argument it cites.
PANEL_DESCRIPTIONS = {
    "space_maps": _description(
        paragraphs=[
            "The source's 1.06 recommendation, verbatim: \"Build co-registered "
            "spaces as the primary architecture and keep overlay always on. "
            "Treat fusion as a named analysis you invoke deliberately, not as "
            'the default view."',
            "Every panel here indexes the same proteins, so a selection in one "
            "is a selection in all of them. Changing the overlay recolours the "
            "points in place; only a different recipe moves them.",
        ],
        sources=["source document 1.06"],
    ),
    "comparisons": _description(
        paragraphs=[
            "Per pair of spaces, how much their neighbourhoods agree. The "
            "per-protein neighbourhood Jaccard behind the means is what the "
            "disagreement button colours by.",
            "A withheld cell is withheld and says why. It is never blank, "
            "because a blank cell reads as a zero.",
        ],
        sources=["coregistration.py", "source document 1.06"],
    ),
    "flavours": _description(
        paragraphs=[
            "The source's 1.02, ordered by how much each option disturbs the "
            "existing geometry. This pipeline implements A (overlay) and F "
            "(co-registered spaces) as its primary architecture, and offers C "
            "(structure + biophysics) as a named fusion.",
        ],
        sources=["source document 1.02"],
    ),
    "signal_inventory": _description(
        paragraphs=[
            "The source's 1.03: \"Some signals belong in the geometry, some "
            "belong only in the legend, and a few must never enter the geometry "
            "because they encode pipeline artifacts, or because they are the "
            'thing your claims get tested against."',
            "This table is read out of the guard that enforces it, so it cannot "
            "drift from what the pipeline actually refuses.",
        ],
        sources=["config_schema.py:103", "docs/adr/0003", "source document 1.03"],
    ),
    "pipeline_diagram": _description(
        paragraphs=[
            "The source's 2.01: \"Read it as a network diagram. Each box is a "
            "tensor, each arrow is an operation, and the line under each box is "
            "what comes out. Two things surprise most people: the input to PCA "
            "is the similarity matrix itself, not a set of protein descriptors, "
            "and clustering runs on a different graph than the one the map is "
            'drawn from."',
            "Drawn from this run's resolved configuration, so it is this "
            "cohort's pipeline and not a diagram of the general case.",
        ],
        sources=["source document 2.01"],
    ),
    "tm_matrix": _description(
        paragraphs=[
            "The source's 2.02: \"Protein i has no feature vector of its own. "
            "Its feature vector is its entire row of TM-scores against every "
            'other protein in the run."',
        ],
        hazards=[
            "The scores are foldseek 3Di+AA alignment scores, not TM-align "
            "output (FOLLOWUPS #60).",
        ],
        sources=["source document 2.02", "blocks/tmscore.py:172"],
    ),
    "censoring": _description(
        paragraphs=[
            "How much of this cohort's all-vs-all comparison was never "
            "measured. A censored cell is one the search never reported, not a "
            "pair that scored zero -- the mask is recovered from the fill "
            "token, because a genuinely measured zero and an unmeasured cell "
            "look identical once both are numbers.",
            "The row/column split is the evidence, not the rate. Foldseek's "
            "`--max-seqs` bounds how many partners each *query* reports, so a "
            "per-query cap piles the rows up at one number while the columns "
            "stay free. A matrix whose rows AND columns both pile up is "
            "uniform for some other reason, and this panel refuses to call "
            "that a cap.",
        ],
        hazards=[
            "Censoring rate is a property of how well a protein was measured, "
            "not of the protein. It correlates with length, which is why it is "
            "overlay-only and may never enter a geometry (ADR 0003).",
            "This panel reports the matrix this cohort was actually built "
            "from. Where a run declares a capped twin it ALSO carries the "
            "comparison against it -- but as numbers, never as a displacement. "
            "Drawing the two layouts side by side is refused on measurement: a "
            "map's censored twin scores 0.968 Procrustes disparity against it, "
            "worse than an entirely different modality, and about thirty times "
            "the reducer's own seed noise. So censoring replaces the frame "
            "rather than moving points, and arrows between the two would invite "
            "exactly the misreading 6.04 warns about (FOLLOWUPS #63).",
        ],
        sources=["matrix_io.py:530", "docs/adr/0009", "docs/adr/0003"],
    ),
    "contributions": _description(
        paragraphs=[
            "The source's 2.04: \"Give every block a weight of 1.0 and you have "
            "not given every block an equal say. What matters is the variance "
            "each block contributes at its own native scale, and those scales "
            'differ by orders of magnitude."',
            "Both numbers are shown because they differ, and the realized one "
            "is the measurement.",
        ],
        sources=["fusion.py:565", "docs/adr/0002", "source document 2.04"],
    ),
    "stability_map": _description(
        paragraphs=[
            "The source's 3.01: \"Before asking what a map means, ask whether "
            'it would look the same if you ran it again."',
        ],
        hazards=[
            "The stability this pipeline computes judges the SPACE, not the "
            "drawn layout (FOLLOWUPS #62), so it does not answer 'would this "
            "point be in the same place'.",
        ],
        sources=["diagnostics/stability.py", "source document 3.01"],
    ),
    "response_profile": _description(
        paragraphs=[
            "The source's 4.01: \"Alanine scanning is the same construct as "
            "occlusion sensitivity in vision: mask a piece of the input, rerun "
            'the model, attribute the change to the piece you masked."',
        ],
        sources=["source document 4.01"],
    ),
    "variant_landing": _description(
        paragraphs=[
            "Where a perturbed variant lands relative to the cluster its wild "
            "type sits in. Leaving the cluster is a testable claim about what "
            "the cluster membership rested on.",
        ],
        sources=["source document 4.01"],
    ),
    "perturbation_grid": _description(
        paragraphs=[
            "The source's 4.02: \"Rows are what you do to the protein, columns "
            "are what you measure afterwards. The empty cells are the point: "
            "most of the grid is uninformative, and the standard instinct lands "
            'in the emptiest corner."',
        ],
        sources=["source document 4.02"],
    ),
    "divergence": _description(
        paragraphs=[
            "Accumulated evolutionary distance on x against structural "
            "divergence on y. The source's x-axis is patristic distance summed "
            "along a gene tree; it is not years, because years need a fossil "
            "calibration.",
        ],
        sources=["source document 5.02"],
    ),
    "innovation_clades": _description(
        paragraphs=[
            "Per group of organisms, how much more or less shape changed than "
            "the branch length predicts. Positive means it changed more than "
            "expected.",
        ],
        sources=["source document 5.03"],
    ),
    "innovation_map": _description(
        paragraphs=[
            "The same residual painted onto the map, because a bar chart cannot "
            "say *where* in shape space the innovation went.",
        ],
        sources=["source document 5.03"],
    ),
    "ancestral_path": _description(
        paragraphs=[
            "Ancestral states placed in the tips' own coordinates and joined "
            "along the branches: not how far a lineage travelled, but which way.",
        ],
        sources=["source document 6.04"],
    ),
    "identity_vs_tm": _description(
        paragraphs=[
            "Sequence identity against structural similarity, pair by pair. "
            "Where sequence has saturated and structure has not, a sequence "
            "tree is reporting alignment jitter.",
        ],
        sources=["source document 6.01"],
    ),
    "records": _description(
        paragraphs=[
            "The proteins themselves: accession, organism, protein name and "
            "chain length, straight from the cohort's UniProt feature table.",
            "Here because every other panel is an aggregate. A map with no way "
            "to ask 'which protein is that point' is a picture, not a result.",
        ],
        sources=["protein_features/uniprot_features.tsv"],
    ),
    "tanglegram": _description(
        paragraphs=[
            "The source's 6.03: \"Straight lines mean the gene history matches "
            'the organism history. Crossings mean something moved."',
        ],
        hazards=[
            "The source's own 6.07 caveat: the crossing count correlates only "
            "weakly with topological distance, so a tanglegram says *which* "
            "tips conflict and never *how much*.",
        ],
        sources=["source document 6.03"],
    ),
    "phylomorphospace": _description(
        paragraphs=[
            "Ancestors as points, branches as routes between them, in the same "
            "coordinates as the tips.",
        ],
        hazards=[
            "The source's 6.07 forbids drawing this on a UMAP: interpolating "
            "between two points assumes the axes are linear combinations of the "
            "data, which is true in PCA and PCoA and false in UMAP.",
        ],
        sources=["source document 6.04"],
    ),
    "tree_space": _description(
        paragraphs=[
            "Every point is an entire phylogeny, placed by tree-to-tree "
            "distance. It answers whether this family's history is typical of "
            "the genome or strange.",
        ],
        sources=["source document 6.05"],
    ),
    "discordance": _description(
        paragraphs=[
            "The source's 6.06: \"Discordance is not one phenomenon. "
            "Duplication, transfer, incomplete lineage sorting and plain "
            "inference error leave different fingerprints, and only some of "
            "them should change how you read the map. The middle column is the "
            "one that matters: it is the test that separates the cause from its "
            'lookalike."',
        ],
        sources=["source document 6.06"],
    ),
    "report": _description(
        paragraphs=[
            "The source's 7.03 E2 fixes this order: cohort and provenance, "
            "retrieval coverage, geometry health, rate fit, and only then the "
            'map. The order is the argument -- "a map with no diagnostics '
            "beside it is exactly the artifact this whole document argues "
            'against".',
        ],
        sources=["source document 7.03 E1-E2"],
    ),
}


def describe_panel(panel_id: str) -> dict:
    """One catalogue panel's fold-out, or an empty one if none is written.

    Empty rather than absent so the template can render the same shape for
    every panel and never has to test for the key.
    """
    return PANEL_DESCRIPTIONS.get(panel_id) or _description()
