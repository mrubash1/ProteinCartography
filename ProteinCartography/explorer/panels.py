#!/usr/bin/env python
"""The catalogue of panels the geometry analysis proposes, and what each needs.

`docs/protein-map-geometry-options-v20.html` argues for a particular explorer
across eight sheets. Most of what it proposes cannot be drawn from what this
pipeline currently produces — it wants a reconciled phylogeny for six panels
and a perturbation scan for three more, and neither exists. The catalogue names
every panel anyway, with the input it is waiting for, for one reason:

    "Refusals are shown as refusals rather than as blanks."  (source, 7.03 E2)

A panel that is silently absent cannot be told apart from one nobody thought
of. A panel that renders its own missing input is a question someone can answer.
So requires is not documentation — it is what the page prints when the data
is not there, and fills_in points at the POST-PLAN entry that would supply it.

Nothing here fetches or computes. This module is deliberately dependency-free
(test_optional_dependencies asserts the explorer imports in a bare env), so it
is a description of panels, and payload.py decides which are satisfiable.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from explorer import descriptions

#: The eight sheets, in the source document's own order. maps is this
#: pipeline's existing grid, which the document calls the primary architecture.
SHEETS = (
    ("maps", "The maps"),
    ("options", "The options"),
    ("mechanics", "How it works"),
    ("stability", "Is it stable"),
    ("perturb", "Perturb and probe"),
    ("time", "Time and innovation"),
    ("trees", "Two trees, one map"),
    ("report", "Diagnostics report"),
)

#: The provenance tag every panel carries, from the source's own four-way
#: split. A reader must never have to guess whether a panel shows measurements
#: or a drawing of an argument.
PROVENANCE = ("real", "mixed", "synthetic", "method")


@dataclass(frozen=True)
class PanelSpec:
    """One proposed panel: what it is, what draws it, what it needs."""

    panel_id: str
    title: str
    sheet: str
    #: Which renderer draws it. "note" and "table" need no numeric input.
    panel_type: str
    #: real / mixed / synthetic / method — see PROVENANCE.
    provenance: str
    #: The source document's own section number, so any claim here is checkable.
    section: str
    #: What the panel answers, in one line, shown under the title.
    question: str = ""
    #: Payload keys that must be present and non-empty for this to draw. Empty
    #: means the panel needs no data and always draws.
    needs: tuple = ()
    #: Printed verbatim when needs is unmet. Name the missing input, not the
    #: fact of its absence — "awaiting a reconciled tree" is actionable and
    #: "no data" is not.
    requires: str = ""
    #: Where in POST-PLAN the work that would fill this panel is described.
    fills_in: str = ""
    #: Free-form content for panels that carry text or rows rather than a plot.
    content: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "panel_id": self.panel_id,
            "title": self.title,
            "sheet": self.sheet,
            "panel_type": self.panel_type,
            "provenance": self.provenance,
            "section": self.section,
            "question": self.question,
            "needs": list(self.needs),
            "requires": self.requires,
            "fills_in": self.fills_in,
            "content": self.content,
        }


#: Inputs this pipeline does not produce, named once so several panels can cite
#: the same one and a reader can see that six panels are blocked on one thing
#: rather than on six different things.
_NO_TREE = (
    "a reconciled gene/species tree. Every cohort this pipeline builds is a "
    "single query's homolog set with no phylogeny attached, and 7.02 correction "
    "2 already refuses the rate ratio when the x-axis is identity-derived "
    "rather than tree-derived — which is the case here"
)
_NO_SCAN = (
    "a perturbation scan: alanine, poly-Ala window, segment and domain "
    "deletion, plus a shuffled control. Roughly 1,900 structure predictions for "
    "two queries, and the predictor choice decides the answer before the "
    "experiment runs, so this is a decision and not only compute"
)


#: The source's 1.02, all six plates, ordered as it orders them: by how much
#: each disturbs the existing geometry. The four prose fields are the source's
#: own words, because the panel's job is to show that argument rather than to
#: paraphrase it.
#:
#: `realized_by` is the interesting column and it is NOT a typed claim. It names
#: the payload key whose presence means THIS RUN actually builds that flavour,
#: so the plate reports the cohort in front of the reader instead of a sentence
#: about the pipeline in general. `unimplemented` is for the three flavours no
#: provider in this pipeline can produce, and it names the missing provider --
#: a test asserts the registry still has none, so the day one is added the claim
#: fails here rather than going stale on the page.
FLAVOUR_PLATES = (
    {
        "plate_id": "A",
        "name": "Color overlay",
        "tag": "Geometry preserved",
        "formula": "geometry = f(TM) · color = g(feature)",
        "answers": (
            "Does fold predict this property in this family? Where do "
            "annotations disagree with structure?"
        ),
        "cannot": (
            "Cannot bring functionally similar, structurally distant proteins "
            "together. Those points are already far apart before you color them."
        ),
        "cost": "Hours. One join onto the features table.",
        "failure": "Reading a spatial pattern into a color that was never in the geometry.",
        "realized_by": "overlays",
    },
    {
        "plate_id": "B",
        "name": "Fusion: structure + sequence",
        "tag": "Geometry altered",
        "formula": "D² = w₁·d²(TM) + w₂·d²(PLM)",
        "answers": (
            "Which proteins are structurally alike but sequence-divergent, and "
            "vice versa. The convergence question."
        ),
        "cannot": (
            "Says nothing about mechanism. Two enzymes with identical fold and "
            "sequence still may not share a substrate."
        ),
        "cost": "Days. ESM-C inference plus a rerun of the reduction.",
        "failure": (
            "The PLM block already encodes structure implicitly, so you "
            "double-count fold and think you fused two views."
        ),
        "unimplemented": "no protein-language-model block provider is registered",
    },
    {
        "plate_id": "C",
        "name": "Fusion: structure + biophysics",
        "tag": "Geometry altered",
        "formula": "D² = w₁·d²(TM) + w₂·d²(GRAVY, charge, aggregation)",
        "answers": (
            "Which structural subfamilies split on surface chemistry. Often the "
            "split that actually matters for expression."
        ),
        "cannot": (
            "Three scalars against an n-dimensional similarity profile. Without "
            "block normalization this is a no-op."
        ),
        "cost": "Days. Descriptors are cheap; the normalization decision is the work.",
        "failure": (
            "Charge in raw units swamps GRAVY in raw units. Standardize within "
            "the block first, then across blocks."
        ),
        "realized_by": "fused_spaces",
        # The source's failure mode for this plate is live in this pipeline, so
        # it is printed on the plate rather than left as a general warning. The
        # block declares `normalization="zscore_within"` and nothing reads that
        # field; late fusion divides each block by its own mean off-diagonal
        # distance, which equalises the BLOCKS and not the columns inside them.
        "hazard": (
            "This pipeline's late fusion normalizes each block by its own mean "
            "off-diagonal distance, which equalises the blocks and not the "
            "columns within them. The biophysical block declares "
            "`normalization: zscore_within` and that field is read by nothing, "
            "so the source's stated failure mode -- one raw scalar swamping "
            "another -- is unguarded here."
        ),
    },
    {
        "plate_id": "D",
        "name": "Fusion: structure + function calls",
        "tag": "Geometry altered",
        "formula": "D² = w₁·d²(TM) + w₂·d²(EC, GO, localization)",
        "answers": (
            "Groups proteins by predicted activity rather than fold, so "
            "pseudo-enzymes separate from enzymes."
        ),
        "cannot": (
            "Inherits predictor error directly into the coordinates. "
            "Best-in-class molecular-function Fmax is around 0.46."
        ),
        "cost": "Days to weeks. CLEAN, DeepFRI, DeepLoc, plus calibration.",
        "failure": (
            "Circularity. Most function predictors were trained on homology, so "
            "you fuse structure with a proxy for structure."
        ),
        "unimplemented": "no function-call block provider is registered",
    },
    {
        "plate_id": "E",
        "name": "Fusion: developability weighted",
        "tag": "Geometry altered",
        "formula": "D² = Σ w·d²(Tm, solubility, MHC-II, humanness)",
        "answers": (
            "Ranks candidates for a build. Neighbors are 'similarly buildable,' "
            "which is what triage needs."
        ),
        "cannot": (
            "Not a biology map. Axes are meaningless and should never be shown "
            "to someone asking an evolutionary question."
        ),
        "cost": (
            "Weeks. Immunogenicity prediction is the long pole and needs "
            "allele-set decisions."
        ),
        "failure": (
            "Shipping it as the exploratory map. Label it as triage or it will "
            "be misread within a week."
        ),
        "unimplemented": "no developability block provider is registered",
    },
    {
        "plate_id": "F",
        "name": "Co-registered spaces",
        "tag": "Parallel geometries",
        "formula": "several maps · one protid index · linked selection",
        "answers": (
            "All of the above, separately, plus the disagreement between them, "
            "which is the actual discovery surface."
        ),
        "cannot": (
            "No single picture to put in a deck. Requires a real interface, not "
            "a static HTML export."
        ),
        "cost": "Weeks. The feature store and the linked-brushing UI are the build.",
        "failure": (
            "Letting the spaces drift out of sync as features are recomputed. "
            "Version the table or the comparison is meaningless."
        ),
        "realized_by": "comparisons",
    },
)


def _flavours() -> dict:
    """The six plates, with their per-run status left for `catalogue_for`."""
    return {
        "plates": [dict(plate) for plate in FLAVOUR_PLATES],
        "note": (
            "The four prose fields are the source's own, at 1.02. Whether a "
            "flavour is built is answered for THIS RUN from what the payload "
            "carries, not from a sentence about the pipeline in general."
        ),
    }


#: What a plate says about itself once the run is known. Three states and never
#: two: a flavour this pipeline cannot build at all is a different fact from one
#: it can build and this cohort did not ask for, and a reader who cannot tell
#: them apart cannot act on either.
PLATE_STATES = ("built by this run", "available, not used by this run", "not implemented")


def _mark_plates(content: dict, available: set) -> dict:
    """Stamp each plate with what this run did about it."""
    plates = content.get("plates")
    if not plates:
        return content
    out = dict(content)
    out["plates"] = []
    for plate in plates:
        plate = dict(plate)
        if plate.get("unimplemented"):
            plate["state"] = "not implemented"
            plate["state_note"] = plate["unimplemented"]
        elif plate.get("realized_by") in available:
            plate["state"] = "built by this run"
            plate["state_note"] = f"the payload carries `{plate['realized_by']}`"
        else:
            plate["state"] = "available, not used by this run"
            plate["state_note"] = (
                f"this run's payload carries no `{plate.get('realized_by')}`, so "
                "nothing here draws it"
            )
        out["plates"].append(plate)
    return out


#: The overlay-only signals, read out of the guard that enforces them.
#:
#: NOT RETYPED, and that is the entire point of the panel. `config_schema`
#: holds the table and rejects a config at parse time; a list typed again here
#: would be free to drift from what the pipeline actually refuses, and a panel
#: showing eight signals while the guard enforced nine would be worse than no
#: panel -- it would look like a check.
#:
#: Grouped by SIGNAL rather than listed by provider, because the guard is keyed
#: on the provider on purpose: `patristic`, `iqtree`, `noveltree` and
#: `phylogeny` are four providers of one circularity, and a table with four
#: rows saying the same sentence hides that.
def _signal_inventory() -> dict:
    from config_schema import NOT_FUSABLE_PROVIDERS, NOT_FUSABLE_REASONS

    by_signal: dict = {}
    for provider, signal in NOT_FUSABLE_PROVIDERS.items():
        by_signal.setdefault(signal, []).append(provider)
    rows = []
    for signal in sorted(by_signal):
        rows.append(
            {
                "signal": signal,
                "providers": ", ".join(sorted(by_signal[signal])),
                # A signal the guard names with no reason recorded is a hole in
                # the argument, not an empty cell. Say which.
                "reason": NOT_FUSABLE_REASONS.get(signal)
                or f"{descriptions.NOT_DETERMINABLE}: no entry in NOT_FUSABLE_REASONS",
            }
        )
    return {
        "columns": [
            {"key": "signal", "label": "signal"},
            {"key": "providers", "label": "providers that produce it"},
            {"key": "reason", "label": "why it may never enter a geometry"},
        ],
        "rows": rows,
        "note": (
            "Read from `config_schema.NOT_FUSABLE_PROVIDERS` when this page was "
            "built, and enforced when a config is parsed -- so a signal added to "
            "the guard appears here without anyone updating the page, and a "
            "signal removed from it disappears."
        ),
    }


#: The five perturbations and five observables of the source's 7.02 grid, and
#: the nine cells it annotates.
#:
#: THE EMPTY CELLS ARE THE CONTENT. The source's argument is that the
#: combination people reach for first -- one alanine against TM-score -- is the
#: emptiest cell in the grid, because a structure predictor leans on the MSA and
#: one substitution barely moves it. A grid drawn without that is a menu; a grid
#: drawn with it is the reason not to spend 126 folding runs.
#:
#: `kind` is a READING of the source's note and the note is printed beside it,
#: so a reader can check the classification rather than take it. Three kinds
#: that must not be collapsed: a cell that works, a cell that is flat by
#: construction, and a control that is not a measurement at all.
#:
#: The sixteen unannotated cells are marked as unannotated and NOT as empty. The
#: source not commenting on a combination is not the source calling it useless,
#: and inventing the difference would be the exact failure this panel is about.
PERTURBATIONS = (
    ("ala1", "Single alanine", "one residue -> Ala, refold each", "126 folds"),
    ("polyA", "Poly-Ala window", "11 residues -> Ala, length preserved", "116 folds"),
    ("segdel", "Segment deletion", "21-residue sliding deletion", "106 folds"),
    ("domdel", "Domain deletion", "remove one domain entirely", "2 folds"),
    ("shuffle", "Shuffled control", "composition matched, order scrambled", "10 folds"),
)

OBSERVABLES = (
    ("tm", "TM-score to WT", "refold, structurally align"),
    ("plddt", "delta mean pLDDT", "predictor confidence shift"),
    ("esm", "PLM log-odds", "masked-marginal against WT"),
    ("disp", "Map displacement", "re-embed, measure movement"),
    ("flip", "Annotation change", "EC / GO / localization flips"),
)

#: cell key -> (kind, the source's own note). Nine of twenty-five.
GRID_CELLS = {
    "ala1|tm": (
        "empty by construction",
        "This is the combination people reach for first and it is the emptiest "
        "cell in the grid. Structure predictors lean on the MSA, and one "
        "substitution barely moves it, so nearly every mutant returns a TM-score "
        "above 0.99. You paid for 126 folding runs and bought a flat line.",
    ),
    "ala1|esm": (
        "informative",
        "The only cell with real single-residue resolution, and it needs no "
        "folding at all. One forward pass gives the whole L x 20 matrix. The "
        "catch: it scores conservation, so a catalytic residue and a buried "
        "structural residue look identical.",
    ),
    "ala1|disp": (
        "empty by construction",
        "Zero by construction. If the structure did not move, the TM-score "
        "profile did not move, so the point does not move. Map displacement can "
        "only respond to perturbations that change the fold.",
    ),
    "polyA|tm": (
        "informative",
        "Eleven alanines is enough to unfold a helix or collapse a loop, so the "
        "structural readout finally lifts off the floor, but only where "
        "secondary structure depended on side chains. Interpret plateaus, not "
        "peaks.",
    ),
    "segdel|tm": (
        "informative",
        "Works, and works well. Deleting 21 residues genuinely changes the fold, "
        "and the profile picks up exactly the regions the protein cannot do "
        "without. This is the honest structural ablation.",
    ),
    "segdel|disp": (
        "informative",
        "The version that answers the map question directly: delete this segment "
        "and the protein leaves its cluster. That is a testable claim about what "
        "the cluster membership was resting on.",
    ),
    "domdel|esm": (
        "empty by construction",
        "Length changed, so per-position likelihoods are not comparable to wild "
        "type. The number will compute and it will mean nothing.",
    ),
    "shuffle|tm": (
        "control",
        "The negative control every ablation panel needs. If a shuffled sequence "
        "still folds to something with a high TM-score to WT, the predictor is "
        "echoing the MSA rather than reading the sequence.",
    ),
    "shuffle|esm": (
        "control",
        "A control, not a measurement. It tells you the floor of the score, "
        "which is the number every real perturbation should be compared against.",
    ),
}

#: The four cell states, which must stay four. Collapsing "the source does not
#: annotate this" into "this is empty" would invent the source's opinion, and
#: collapsing a control into a measurement is how a noise floor gets quoted as
#: a result.
CELL_KINDS = ("informative", "empty by construction", "control", "unannotated")


def _perturbation_grid() -> dict:
    cells = {}
    for pert_key, _, _, _ in PERTURBATIONS:
        for obs_key, _, _ in OBSERVABLES:
            key = f"{pert_key}|{obs_key}"
            kind, note = GRID_CELLS.get(
                key, ("unannotated", "The source does not annotate this combination.")
            )
            cells[key] = {"kind": kind, "note": note}
    return {
        "perturbations": [
            {"key": key, "name": name, "sub": sub, "cost": cost}
            for key, name, sub, cost in PERTURBATIONS
        ],
        "observables": [
            {"key": key, "name": name, "sub": sub} for key, name, sub in OBSERVABLES
        ],
        "cells": cells,
        "kinds": list(CELL_KINDS),
        "note": (
            "No scan has been run: these are the twenty-five combinations, not "
            "results. The point of the grid is which cells are flat before you "
            "pay for them -- one alanine against TM-score is the emptiest cell "
            "and the one most people run first. Each classification is a reading "
            "of the source's note, which is printed with it."
        ),
    }


#: The seven discordance patterns of the source's 6.06, verbatim.
#:
#: Discordance is not one phenomenon, and the source's point is that the TEST
#: column is the one that matters: it is what separates a cause from its
#: lookalike. Two of the seven -- saturation and structural noise -- resolve in
#: OPPOSITE directions on which tree to trust, so a table listing causes without
#: their tests would let a reader pick whichever conclusion they were hoping for.
#:
#: Static, and it stays static. Nothing in this pipeline produces a gene tree, so
#: none of these patterns can be detected here; the panel is a reading guide for
#: the day a tree arrives, which is why its provenance tag is `method` and not
#: `real`.
DISCORDANCE_PATTERNS = (
    (
        "Two copies per species, sister to each other",
        "Recent duplication",
        "Duplication node sits below the speciation node it should sit above; "
        "both copies present in most descendants",
        "Paralogs land close together; a tight pair that is not a fold discovery",
    ),
    (
        "Two copies, each sister to the other species' copy",
        "Ancient duplication then loss",
        "Copies group by paralog class, not by species; the split predates the taxa",
        "Two parallel clusters offset in the same direction; the offset is the "
        "functional divergence",
    ),
    (
        "One tip nested deep inside an unrelated clade",
        "Horizontal transfer",
        "Short patristic distance, wildly wrong taxonomy, often a different GC or "
        "codon bias; screen with PreHGT",
        "A single point far from its taxonomic neighbours. This is the outlier "
        "people misread as innovation",
    ),
    (
        "Shallow relationships shuffled, deep ones stable",
        "Incomplete lineage sorting",
        "Concentrated on short internal branches; conflicting topologies appear at "
        "roughly equal frequency",
        "Almost none. ILS moves the tree, not the structure, which is exactly why "
        "it is safe to ignore here",
    ),
    (
        "Long branch attaching to another long branch",
        "Long-branch attraction",
        "Goes away under a better model or with more taxa; bootstrap high but "
        "posterior unstable",
        "Fake innovation: the rate looks enormous because the branch length is wrong",
    ),
    (
        "Structure tree and gene tree disagree at deep nodes",
        "Sequence saturation",
        "Sequence identity below ~20% across the split; structure still carries signal",
        "Trust the structure tree here. This is the regime the map exists for",
    ),
    (
        "Structure tree and gene tree disagree at shallow nodes",
        "Structural noise or model error",
        "pLDDT low in the regions driving the TM difference; check per-residue "
        "confidence",
        "Trust the gene tree here. Points may be adjacent for prediction-quality "
        "reasons only",
    ),
)


def _discordance() -> dict:
    return {
        "columns": [
            {"key": "seen", "label": "what you see"},
            {"key": "cause", "label": "most likely cause"},
            {"key": "test", "label": "how to tell it apart"},
            {"key": "effect", "label": "effect on the map"},
        ],
        "rows": [
            {"seen": seen, "cause": cause, "test": test, "effect": effect}
            for seen, cause, test, effect in DISCORDANCE_PATTERNS
        ],
        "note": (
            "The middle column is the one that matters: it is the test that "
            "separates a cause from its lookalike. Two of the seven resolve in "
            "opposite directions on which tree to trust, so a cause read without "
            "its test is a conclusion chosen rather than measured. Nothing in "
            "this pipeline produces a gene tree, so none of these can be detected "
            "here -- this is a reading guide, not a measurement of this cohort."
        ),
    }


#: The report's sections, in the order 7.03 E2 fixes: cohort and provenance,
#: retrieval coverage, geometry health, rate fit, and only THEN the map.
#:
#: A tuple and not a mapping, deliberately. The template renders these in list
#: order rather than looking each one up by name, so the order is data that
#: travels to the page and can be asserted on here. A template free to look
#: sections up is a template free to put the map first, which is the one thing
#: the source says a report must not do.
#:
#: `refused` is what a section prints when the payload does not carry its
#: input. It names the input and where the input lives, because a refusal a
#: reader can act on is the whole difference between this and a blank.
REPORT_SECTIONS = (
    {
        "section_id": "cohort",
        "title": "1. Cohort and provenance",
        "lead": "Which proteins, chosen by which rule, and from which cache keys.",
    },
    {
        "section_id": "coverage",
        "title": "2. Retrieval coverage",
        "lead": "How much of the all-vs-all comparison was actually measured.",
        "refused": (
            "the explorer does not read the retrieval stage. The per-query cap's "
            "censored fraction is computed and written to the tmscore block's "
            "manifest, and no payload key carries it -- so this section refuses "
            "rather than report a coverage of 100%"
        ),
        "fills_in": (
            "the censoring panel's input: both matrices for one cohort, plumbed " "into the payload"
        ),
    },
    {
        "section_id": "geometry",
        "title": "3. Geometry health",
        "lead": (
            "Per space: neighbourhood stability, layout faithfulness, the "
            "partition, and whether the clusters beat their negative control."
        ),
    },
    {
        "section_id": "rate",
        "title": "4. Rate fit",
        "lead": "How much structural change a unit of evolutionary distance buys.",
        "refused": (
            "there is no evolutionary distance to fit against. Every cohort this "
            "pipeline builds is one query's homolog set with no phylogeny "
            "attached, and the rate ratio is refused outright when the x-axis is "
            "identity-derived rather than tree-derived, which is the case for "
            "every cohort this pipeline can currently build"
        ),
        "fills_in": (
            "a phylogeny: patristic distance per protein, tip labels matching "
            "the cohort's protids"
        ),
    },
    {
        "section_id": "map",
        "title": "5. The map, last",
        "lead": (
            "What each panel may be read for. Fifth and not first, which is the "
            "whole argument of the section this order comes from."
        ),
    },
)


CATALOGUE = (
    # --- the maps, which this pipeline already draws --------------------------
    PanelSpec(
        panel_id="space_maps",
        title="Co-registered spaces",
        sheet="maps",
        panel_type="space_grid",
        provenance="real",
        section="1.06",
        question=(
            "Do these blocks tell the same story? Overlay recolours in place; "
            "only a change of recipe moves a point."
        ),
    ),
    PanelSpec(
        panel_id="comparisons",
        title="Cross-space disagreement",
        sheet="maps",
        panel_type="comparisons",
        provenance="real",
        section="1.06",
        question="Where do two spaces put the same protein in different company?",
        needs=("comparisons",),
        requires=(
            "a coregistration/summary.tsv. It is written by coregister.py, "
            "which needs at least two spaces sharing a protid index"
        ),
        fills_in="run coregister.py for the cohort",
    ),
    # --- the options ----------------------------------------------------------
    PanelSpec(
        panel_id="flavours",
        title="Six ways to build a map",
        sheet="options",
        panel_type="cards",
        provenance="method",
        section="1.02",
        question="Which of these are you actually asking for?",
        content=_flavours(),
    ),
    PanelSpec(
        panel_id="signal_inventory",
        title="Overlay yes, fusion no",
        sheet="options",
        panel_type="table",
        provenance="method",
        section="1.03",
        question=(
            "Eight quantities that may colour a map and must never enter its "
            "geometry. Enforced, not merely listed — see the geometry guard."
        ),
        content=_signal_inventory(),
    ),
    # --- how it works ---------------------------------------------------------
    PanelSpec(
        panel_id="pipeline_diagram",
        title="What the pipeline actually is",
        sheet="mechanics",
        panel_type="pipeline",
        provenance="method",
        section="2.01",
        question="Where does a second modality join, and on which graph does each step run?",
    ),
    PanelSpec(
        panel_id="tm_matrix",
        title="The similarity matrix",
        sheet="mechanics",
        panel_type="heatmap",
        provenance="real",
        section="2.02",
        question="A row of this matrix IS the feature vector. Sorted by cluster.",
        needs=("matrix",),
        requires="the cohort's all-vs-all TM matrix, as an n x n labelled TSV",
        fills_in="point the explorer at the cohort's matrix.tsv",
    ),
    PanelSpec(
        panel_id="censoring",
        title="What the per-query cap removes",
        sheet="mechanics",
        panel_type="censoring",
        provenance="real",
        section="2.02",
        question=(
            "Open rings are the uncensored position; the line is how far a "
            "protein moves once the cap decides who measured whom."
        ),
        needs=("censoring",),
        requires=(
            "both matrices for one cohort — the shipped capped one and an "
            "exhaustive one. N7 produced the second for actin_B and chymo_A1"
        ),
        fills_in="POST-PLAN, N7: the exhaustive matrices are on disk",
    ),
    PanelSpec(
        panel_id="contributions",
        title="Why an equal weight is not an equal say",
        sheet="mechanics",
        panel_type="contributions",
        provenance="real",
        section="2.04",
        question="Effective contribution to squared distance, per block.",
        needs=("fused_spaces",),
        requires="at least one space with a fusion strategy and two or more blocks",
        fills_in="the cohort configs define fused_late",
    ),
    # --- is it stable ---------------------------------------------------------
    PanelSpec(
        panel_id="stability_map",
        title="Would you get the same neighbours again",
        sheet="stability",
        panel_type="stability",
        provenance="real",
        section="3.01",
        question=(
            "The source calls this overlay not optional. Note it currently "
            "measures the SPACE, not the layout — FOLLOWUPS #62."
        ),
        needs=("stability_series",),
        requires=(
            "a per-protein stability series. diagnostics/stability.py computes "
            "one and diagnostics.json exports only the coin-flip list, the "
            "mean and the min, so the ramp this panel needs is discarded"
        ),
        fills_in="POST-PLAN, the explorer build-out: per-protein stability export",
    ),
    # --- perturb and probe ----------------------------------------------------
    PanelSpec(
        panel_id="response_profile",
        title="Response profile, per residue",
        sheet="perturb",
        panel_type="profile",
        provenance="synthetic",
        section="4.01",
        question="Which parts of the chain hold the protein in position?",
        needs=("perturbation",),
        requires=_NO_SCAN,
        fills_in="GEOMETRY_DIGEST 0.04 decision 3 — Matt's call, not a build task",
    ),
    PanelSpec(
        panel_id="variant_landing",
        title="Where the variants land",
        sheet="perturb",
        panel_type="scatter_overlay",
        provenance="synthetic",
        section="4.01",
        question="Does a variant leave its cluster?",
        needs=("perturbation",),
        requires=_NO_SCAN,
        fills_in="GEOMETRY_DIGEST 0.04 decision 3",
    ),
    PanelSpec(
        panel_id="perturbation_grid",
        title="Perturbation against observable",
        sheet="perturb",
        panel_type="grid",
        provenance="method",
        section="4.02",
        question=(
            "Five perturbations against five observables. The source's point is "
            "that the EMPTY CELLS are the content — this grid is not a stub."
        ),
        content=_perturbation_grid(),
    ),
    # --- time and innovation --------------------------------------------------
    PanelSpec(
        panel_id="divergence",
        title="Divergence against evolutionary distance",
        sheet="time",
        panel_type="scatter_fit",
        provenance="real",
        section="5.02",
        question="How much change should a branch length buy?",
        needs=("patristic",),
        requires=_NO_TREE,
        fills_in="PLAN Phase 9 item 3, and GEOMETRY_DIGEST correction 2",
    ),
    PanelSpec(
        panel_id="innovation_clades",
        title="Innovation by clade",
        sheet="time",
        panel_type="bars",
        provenance="real",
        section="5.03",
        question="Which group changed more than its branch length predicts?",
        needs=("patristic", "clades"),
        requires=_NO_TREE,
        fills_in="PLAN Phase 9 item 3",
    ),
    PanelSpec(
        panel_id="innovation_map",
        title="Innovation, on the map",
        sheet="time",
        panel_type="scatter_overlay",
        provenance="real",
        section="5.03",
        question="Where in shape space the innovation went, which a bar chart cannot show.",
        needs=("patristic",),
        requires=_NO_TREE,
        fills_in="PLAN Phase 9 item 3",
    ),
    PanelSpec(
        panel_id="ancestral_path",
        title="The route a lineage took",
        sheet="time",
        panel_type="trajectory",
        provenance="mixed",
        section="5.03",
        question="Not how far a lineage travelled, but which way.",
        needs=("ancestors",),
        requires="ancestral state reconstruction, which needs " + _NO_TREE,
        fills_in="PLAN Phase 9",
    ),
    # --- two trees, one map ---------------------------------------------------
    PanelSpec(
        panel_id="identity_vs_tm",
        title="Sequence identity against structural similarity",
        sheet="trees",
        panel_type="scatter",
        provenance="real",
        section="6.01",
        question="Can this family carry a tree at all?",
        needs=("identity",),
        requires=(
            "a sequence-identity table. This is the cheapest gap on the page: "
            "aggregate_foldseek_fraction_seq_identity.py already exists and "
            "foldseek emits fident, so it is one search pass per cohort"
        ),
        fills_in="POST-PLAN, the explorer build-out: the cheapest empty to fill",
    ),
    PanelSpec(
        panel_id="records",
        title="What these proteins are",
        sheet="trees",
        panel_type="records",
        provenance="real",
        section="6.01",
        question="Accession, organism, protein, length.",
        needs=("records",),
        requires="the cohort's uniprot_features.tsv",
        fills_in="present in both cohort trees",
    ),
    PanelSpec(
        panel_id="tanglegram",
        title="Gene history against organism history",
        sheet="trees",
        panel_type="tanglegram",
        provenance="real",
        section="6.03",
        question="Where does gene history disagree with organism history?",
        needs=("gene_tree", "species_tree"),
        requires=_NO_TREE,
        fills_in="PLAN Phase 9",
    ),
    PanelSpec(
        panel_id="phylomorphospace",
        title="The tree, drawn into the map",
        sheet="trees",
        panel_type="phylomorphospace",
        provenance="mixed",
        section="6.04",
        question=(
            "Ancestors as points, branches as routes. Requires a METRIC "
            "embedding — 6.04 forbids drawing branches on a UMAP."
        ),
        needs=("gene_tree", "metric_embedding"),
        requires=_NO_TREE,
        fills_in="PLAN Phase 9",
    ),
    PanelSpec(
        panel_id="tree_space",
        title="Is this family a typical member of the genome",
        sheet="trees",
        panel_type="scatter",
        provenance="synthetic",
        section="6.05",
        question="Every point is an entire phylogeny.",
        needs=("tree_corpus",),
        requires="a corpus of per-family trees; the source demonstrates with 122 families",
        fills_in="PLAN Phase 9",
    ),
    PanelSpec(
        panel_id="discordance",
        title="Seven ways two trees disagree",
        sheet="trees",
        panel_type="table",
        provenance="method",
        section="6.06",
        question="The cause, the test that separates it from its lookalike, and the effect.",
        content=_discordance(),
    ),
    # --- the report, whose section order is fixed by the source ---------------
    PanelSpec(
        panel_id="report",
        title="Diagnostics report",
        sheet="report",
        panel_type="report",
        provenance="real",
        section="7.03",
        question=(
            "Cohort and provenance, retrieval coverage, geometry health, rate "
            "fit, and only THEN the map. The order is the source's, and it is "
            "the point: a map with no diagnostics beside it is the artifact the "
            "whole document argues against."
        ),
        content={"sections": list(REPORT_SECTIONS)},
    ),
)


def sheet_titles() -> list:
    """The sheets, in order, as [{id, title}]."""
    return [{"sheet": s, "title": t} for s, t in SHEETS]


def catalogue_for(available: set) -> list:
    """Every panel, each marked drawable or awaiting, given what the payload has.

    Returns dicts rather than PanelSpecs because this goes straight into the
    page. drawable is computed here, once, so the template never has to decide
    whether a panel has data — a decision split across two languages is how a
    panel ends up blank in one of them.
    """
    out = []
    for spec in CATALOGUE:
        missing = [name for name in spec.needs if name not in available]
        entry = spec.to_dict()
        entry["drawable"] = not missing
        entry["missing"] = missing
        # The fold-out. Attached here rather than stored on the PanelSpec so
        # that the catalogue stays a description of panels and the prose stays
        # in one file where it can be checked against the code it describes.
        entry["description"] = descriptions.describe_panel(spec.panel_id)
        entry["content"] = _mark_plates(entry["content"], available)
        out.append(entry)
    return out
