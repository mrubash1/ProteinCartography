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
        out.append(entry)
    return out
