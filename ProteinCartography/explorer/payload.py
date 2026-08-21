#!/usr/bin/env python
"""Assemble everything the explorer draws, from a finished run directory.

Separated from the HTML so that what the explorer *says* can be tested without
parsing a web page. Every judgement about whether a space is readable is made
here and travels to the browser as a flag; the template renders flags and does
not decide anything.

**The design constraint that shaped this file.** Nine diagnostics exist and
several of them say *do not read this* -- `stability.informative` is false for
all seven spaces of the shipped demo, `partition.caveat` says when a space's
clusters are really another space's, and `faithfulness` flags individual
proteins whose 2-D position is not meaningful. An explorer that renders those
identically to a trustworthy space undoes the group that produced them, so the
payload carries three levels of refusal:

* a **space-level** verdict, so a panel can be banded and captioned;
* a **protein-level** mask, so an unreadable point is drawn differently from a
  readable one in the same panel;
* a **comparison-level** reason, so a withheld number renders as "withheld,
  because ..." rather than as a blank cell that reads like a zero.

Nothing here invents a threshold. Every band comes from the module that
computes the statistic -- `diagnostics.embedding.FAITHFUL_THRESHOLD`,
`diagnostics.stability.COIN_FLIP_THRESHOLD` and so on -- so the explorer and
`docs/INTERPRETING.md` cannot drift apart.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field

from explorer import descriptions, panels
from spaces import layout

__all__ = [
    "SpacePayload",
    "ExplorerPayload",
    "build_payload",
    "read_embedding",
    "space_verdict",
]

#: Only these keys of a diagnostics report travel to the browser. The report
#: also carries per-protein tables that would multiply the file size for
#: something no panel draws.
SUMMARY_SECTIONS = (
    "redundancy",
    "negative_controls",
    "resolution_sweep",
    "partition",
    "censoring",
)

#: Keys stripped out of a `censoring` entry before it travels. The section is
#: the one diagnostic written PER SPACE about the matrix a space was built from,
#: and it is the only one carrying a table rather than a summary:
#: `cross_cluster_table` is one row per ORDERED cluster pair, so it grows as the
#: square of the cluster count -- 49 rows at k=7, 64 at k=8, 36 at k=6 on the
#: two shipped cohorts -- and no panel draws it. Its reduction,
#: `cross_cluster_edge_retention`, is what the page reads, and it is six numbers.
#:
#: Named per key rather than by an allow-list on purpose: a NEW key added to the
#: censoring diagnostic should reach the browser by default and be noticed, not
#: be silently dropped by a list nobody updated.
CENSORING_DROPPED_KEYS = ("cross_cluster_table",)


def read_embedding(path: str) -> tuple:
    """``(protids, [[x, y], ...])`` from a reducer's TSV, read by label."""
    import pandas as pd

    frame = pd.read_csv(path, sep="\t", index_col=0)
    coordinates = frame.iloc[:, :2].to_numpy(dtype=float)
    return [str(p) for p in frame.index], [[float(x), float(y)] for x, y in coordinates]


def _read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def _read_clusters(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    import pandas as pd

    frame = pd.read_csv(path, sep="\t")
    if "protid" not in frame.columns or frame.shape[1] < 2:
        return {}
    label = next(c for c in frame.columns if c != "protid")
    return {str(p): str(v) for p, v in zip(frame["protid"], frame[label])}


@dataclass(frozen=True)
class SpacePayload:
    """One panel's data, and what it may be read for."""

    space_id: str
    protids: list
    #: reducer name -> [[x, y], ...] in ``protids`` order
    embeddings: dict
    clusters: dict
    #: Per protein: True when its 2-D position is worth reading. Derived from
    #: `faithfulness`, which is per protein and per reducer; a protein is
    #: unreadable if *any* reducer flags it, because the panel shows one.
    readable: list
    verdict: dict
    diagnostics: dict = field(default_factory=dict)
    #: Per-block nominal and realized contribution, for fused spaces only.
    #: Empty for an unfused space, which has nothing to apportion.
    contributions: list = field(default_factory=list)
    #: Which renderer draws this panel. The template dispatches on it. Defaults
    #: to the only kind that existed before the registry, so a payload written
    #: without it renders exactly as it always did -- and an unknown value is
    #: reported on the page rather than silently dropped, because a missing
    #: panel and a broken panel look identical to a reader.
    panel_type: str = "scatter"
    #: What this map is plotting, in words: ``{paragraphs, hazards, sources}``
    #: from `explorer.descriptions`. Empty for a payload built before the
    #: fold-outs existed, and the template renders nothing for an empty one.
    description: dict = field(default_factory=dict)
    #: ``{protid: neighborhood_stability}`` for this space. Per space rather
    #: than per reducer, because that is what the measurement is.
    stability: dict = field(default_factory=dict)
    #: Per named column, its share of this space's squared distance. Empty
    #: unless the space is one feature block whose columns are named.
    column_shares: list = field(default_factory=list)
    #: ``{reducer_id: {axis_names, pca_components_requested,
    #: pca_components_used}}``, read from each reducer's own manifest. The page
    #: opens on one reducer but can switch, and the axes sentence differs
    #: between them, so this travels per reducer rather than resolved.
    axes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "space_id": self.space_id,
            "protids": self.protids,
            "embeddings": self.embeddings,
            "clusters": self.clusters,
            "readable": self.readable,
            "verdict": self.verdict,
            "diagnostics": self.diagnostics,
            "contributions": self.contributions,
            "panel_type": self.panel_type,
            "description": self.description,
            "stability": self.stability,
            "column_shares": self.column_shares,
            "axes": self.axes,
        }


def space_verdict(diagnostics, n_proteins: int) -> dict:
    """What this space may be read for, as flags a template can render.

    Returns ``level`` in ``{"ok", "caution", "unreadable"}`` plus the reasons.
    The levels are deliberately coarse: a panel can be bordered three ways and
    a reader can hold three categories, and a finer scale would imply a
    precision these diagnostics do not have.
    """
    from diagnostics.embedding import DISTORTED_THRESHOLD
    from diagnostics.stability import COIN_FLIP_THRESHOLD

    reasons, level = [], "ok"
    if not diagnostics:
        return {
            "level": "caution",
            "reasons": ["no diagnostics were written for this space, so nothing here is qualified"],
            "headline": "undiagnosed",
        }

    stability = (diagnostics.get("stability") or [{}])[0]
    if stability and not stability.get("informative", True):
        level = "unreadable"
        reasons.append(
            f"stability is not informative: k={stability.get('k')} of "
            f"{max(1, stability.get('subsample_size', 1)) - 1} candidates a replicate "
            "offers, so every protein's neighbours are most of the cohort and the "
            "score is 1.0 by construction"
        )
    elif stability and stability.get("stability_mean", 1.0) <= COIN_FLIP_THRESHOLD:
        level = "unreadable"
        reasons.append(
            # THREE decimals, matching `diagnostics.stability`'s own sentence
            # exactly. At two the banner said 0.12 while the diagnostic three
            # lines below it said 0.119 about the same measurement, and phase 3
            # puts the two on one screen -- one number printed twice at two
            # precisions reads as two measurements that disagree.
            f"mean neighbourhood stability is {stability['stability_mean']:.3f}, at or "
            "below the coin-flip level"
        )

    # A report with sections but NO faithfulness section is a report where
    # nothing looked at the 2-D coordinates. That is not the same as a layout
    # with no problems, and reading it as one is the #29/#32 shape again --
    # absence taken for absence-of-problem. On any pipeline run this cannot
    # fire, because the Snakefile always passes `--embedding`; it fires exactly
    # when someone regenerated a report without it, which is when a reader most
    # needs telling.
    if "faithfulness" not in diagnostics:
        level = "unreadable" if level == "unreadable" else "caution"
        reasons.append(
            "no faithfulness section was written for this space, so nothing has "
            "checked whether its 2-D layout preserves the neighbourhoods it came "
            "from. The positions here are undiagnosed rather than sound"
        )

    for entry in diagnostics.get("faithfulness") or []:
        mean_t = entry.get("trustworthiness_mean")
        mean_c = entry.get("continuity_mean")
        if mean_t is None or mean_c is None:
            continue
        if min(mean_t, mean_c) <= DISTORTED_THRESHOLD:
            level = "unreadable" if level == "unreadable" else "caution"
            reasons.append(
                f"the {entry.get('reducer')} layout is substantially distorted: "
                f"trustworthiness {mean_t:.2f}, continuity {mean_c:.2f}"
            )

    partition = diagnostics.get("partition") or {}
    if partition.get("caveat"):
        level = "unreadable" if level == "unreadable" else "caution"
        reasons.append(partition["caveat"])

    controls = diagnostics.get("negative_controls") or {}
    for name, margin in (controls.get("margins") or {}).items():
        if margin <= 0:
            level = "unreadable"
            reasons.append(
                f"this space's clusters are no better separated than the '{name}' control"
            )

    headline = {
        "ok": "diagnostics found no reason to distrust this map",
        "caution": "read with the caveats below",
        "unreadable": "this map should not be read as a finding",
    }[level]
    return {"level": level, "reasons": reasons, "headline": headline}


def _readable_mask(directory: str, diagnostics, protids: list) -> list:
    """Per protein, whether its position is worth reading.

    A protein is unreadable when any reducer's trustworthiness *or* continuity
    for it falls at or below the distorted band. The two are never averaged --
    they fail in opposite directions, and a point that is in the wrong
    neighbourhood is as unreadable as one torn away from its own.

    Read from ``faithfulness_{reducer}.tsv`` rather than from
    ``diagnostics.json``. The first draft read per-protein arrays out of the
    JSON; those keys do not exist -- `EmbeddingFaithfulness.to_dict` carries
    only the means, and the per-protein values go to the TSV -- so every
    protein came back readable and the whole mechanism was silently inert. It
    passed every shape check. Running it against the demo is what showed the
    mask was uniformly True on a cohort where the diagnostics flag most points.
    """
    import pandas as pd
    from diagnostics.embedding import DISTORTED_THRESHOLD

    readable = dict.fromkeys(protids, True)
    seen_any = False
    for entry in (diagnostics or {}).get("faithfulness") or []:
        path = os.path.join(directory, layout.faithfulness_filename(entry.get("reducer")))
        if not os.path.exists(path):
            continue
        seen_any = True
        frame = pd.read_csv(path, sep="\t")
        if not {"protid", "trustworthiness", "continuity"} <= set(frame.columns):
            continue
        for name, trust, cont in zip(
            frame["protid"], frame["trustworthiness"], frame["continuity"]
        ):
            if min(float(trust), float(cont)) <= DISTORTED_THRESHOLD:
                readable[str(name)] = False
    if (diagnostics or {}).get("faithfulness") and not seen_any:
        # The section exists and none of its tables do. Better to mark the whole
        # space unreadable than to draw every point as trustworthy on the
        # strength of a file that is missing.
        return [False] * len(protids)
    return [bool(readable.get(p, True)) for p in protids]


#: The columns the records panel shows, and what it calls them. Four, because
#: the panel answers "which protein is that point" and nothing else; the feature
#: table has 23 columns and shipping all of them would be a database export
#: rather than a panel.
RECORD_COLUMNS = (
    ("protid", "accession"),
    ("Protein names", "protein"),
    ("Organism", "organism"),
    ("Length", "length"),
)


def _records(output_dir: str, protids: list) -> list:
    """Accession, protein, organism and length, in the panel's own order.

    Read from `protein_features/uniprot_features.tsv`, which is what the
    catalogue entry says it requires -- deliberately not from the aggregated
    features table. Overlays come from the aggregated table and are filtered
    down to what can be a colour, so `Protein names` and `Organism` are dropped
    from it for having too many levels. Those two are most of the point here.

    Restricted to the proteins that are actually on the maps. A record for a
    protein no panel plots would be a row a reader cannot find, and it would
    make the table disagree with the protein count in the footer.
    """
    path = os.path.join(output_dir, "protein_features", "uniprot_features.tsv")
    if not os.path.exists(path):
        return []
    import pandas as pd

    frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if "protid" not in frame.columns:
        return []
    wanted = set(protids)
    rows = []
    for record in frame.to_dict("records"):
        protid = str(record.get("protid", ""))
        if protid not in wanted:
            continue
        rows.append({name: str(record.get(column, "")) for column, name in RECORD_COLUMNS})
    return rows


def _stability_series(directory: str, protids: list) -> dict:
    """`{protid: neighborhood_stability}` for one space, or `{}`.

    Per SPACE and not per reducer: stability is measured on the space's own
    distances, so it says whether these proteins have determinate neighbours at
    all, and NOT whether a particular layout drew them faithfully (FOLLOWUPS
    #62). The two are separate judgements and the panel says which this is.
    """
    path = os.path.join(directory, layout.stability_filename())
    if not os.path.exists(path):
        return {}
    import pandas as pd

    frame = pd.read_csv(path, sep="\t")
    if "protid" not in frame or "stability" not in frame:
        return {}
    wanted = set(protids)
    pairs = zip(frame["protid"], frame["stability"])
    return {
        str(protid): float(value)
        for protid, value in pairs
        if str(protid) in wanted and not pd.isna(value)
    }


def _space_blocks(output_dir: str, config, space) -> list:
    """Per block of one space: provider, resolved params, and its cohort facts.

    The facts are the block manifest's ``extra``, and reading them is what keeps
    the fold-outs honest across cohorts. The 3Di vocabulary is 4982 columns wide
    for one cohort on the shipped page and 4594 for the other; a description
    that stated either number as a constant would be a measurement-looking
    sentence that is wrong on one of the two panels it appears in.

    Nothing from here reaches the payload. `explorer.descriptions` turns it into
    strings and only the strings travel, so a manifest key that happens to hold
    a 5000-element vocabulary list cannot land in the page.
    """
    out = []
    for block_id in getattr(space, "blocks", ()) or ():
        block = (getattr(config, "blocks", None) or {}).get(block_id)
        params = dict(getattr(block, "params", {}) or {})
        # `representation`, `metric` and `normalization` are fields of
        # `BlockConfig` rather than entries in `params`, so a describer reading
        # only `params` would report every tmscore block as the default.
        for name in ("representation", "metric", "normalization"):
            value = getattr(block, name, None)
            if value is not None:
                params.setdefault(name, value)
        manifest = _read_json(layout.block_manifest_path(output_dir, block_id)) or {}
        out.append(
            {
                "block_id": block_id,
                "provider": getattr(block, "provider", "") or "",
                "params": params,
                "facts": dict(manifest.get("extra") or {}),
            }
        )
    return out


def _hover_fields(output_dir: str, protids: set) -> dict:
    """Per protein, the two things a reader recognises: species and Leiden cluster.

    Deliberately NOT sourced from the overlay table. Overlays come from
    `final_results/*_aggregated_features.tsv`, which a run only has if
    `aggregate_features` ran; neither cohort here has one, so every overlay is
    absent and the tooltip would be an accession and nothing else. These two
    come from files every run writes.

    Species is `Organism` from the protein feature table. The Leiden cluster is
    the whole-cohort partition from the TM matrix -- the one the published maps
    are labelled by -- and is distinct from a space's own `clusters.tsv`, which
    is why the tooltip names them differently.
    """
    import pandas as pd

    out: dict = {}
    features = os.path.join(output_dir, "protein_features", "uniprot_features.tsv")
    if os.path.exists(features):
        frame = pd.read_csv(features, sep="\t", dtype=str).fillna("")
        if "protid" in frame.columns and "Organism" in frame.columns:
            for protid, organism in zip(frame["protid"], frame["Organism"]):
                if str(protid) in protids and organism:
                    out.setdefault(str(protid), {})["species"] = str(organism)

    leiden = os.path.join(output_dir, "foldseek_clustering_results", "leiden_features.tsv")
    if os.path.exists(leiden):
        frame = pd.read_csv(leiden, sep="\t", dtype=str).fillna("")
        if "protid" in frame.columns:
            column = next((c for c in frame.columns if c != "protid"), None)
            if column:
                for protid, value in zip(frame["protid"], frame[column]):
                    if str(protid) in protids and value != "":
                        out.setdefault(str(protid), {})["leiden"] = str(value)
    return out


@dataclass(frozen=True)
class ExplorerPayload:
    """Everything the single HTML file embeds."""

    analysis_name: str
    spaces: list
    comparisons: list
    overlays: dict
    provenance: dict
    #: Every panel the geometry analysis proposes, each already marked drawable
    #: or awaiting a named input. Computed here rather than in the template: a
    #: decision split across two languages is how a panel ends up blank in one
    #: of them.
    panels: list = field(default_factory=list)
    #: The sheets those panels are grouped under, in the source's order.
    sheets: list = field(default_factory=list)
    #: protid -> {species, leiden}. What the cursor shows beyond the accession.
    hover: dict = field(default_factory=dict)
    #: One row per protein on the maps: accession, protein, organism, length.
    #: Empty for a run with no UniProt feature table, in which case the records
    #: panel says which file it is waiting for.
    records: list = field(default_factory=list)
    #: The resolved pipeline as boxes and arrows: the blocks, and per space the
    #: reduction and the SEPARATE graph Leiden clusters on. Empty for a config
    #: that resolved to neither, in which case the panel says so.
    pipeline: dict = field(default_factory=dict)
    #: The cohort's similarity matrix, cluster-sorted and quantised to uint8 for
    #: display, with the asymmetry it carries measured beside it. Empty when the
    #: config points at no matrix.
    tm_matrix: dict = field(default_factory=dict)
    #: What the per-query cap removed, from `matrix_io.summarize_censoring`.
    #: Carries its zero for an uncensored matrix rather than being empty.
    censoring: dict = field(default_factory=dict)
    #: The judgement lines the page draws, read from the modules that enforce
    #: them rather than retyped into the template.
    thresholds: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "analysis_name": self.analysis_name,
            "spaces": [space.to_dict() for space in self.spaces],
            "comparisons": self.comparisons,
            "overlays": self.overlays,
            "provenance": self.provenance,
            "panels": self.panels,
            "sheets": self.sheets,
            "hover": self.hover,
            "records": self.records,
            "pipeline": self.pipeline,
            "tm_matrix": self.tm_matrix,
            "censoring": self.censoring,
            "thresholds": self.thresholds,
        }


def _overlays_from_features(path: str, protids: list) -> dict:
    """Columns from `aggregated_features.tsv` a reader may color by.

    Reused rather than reinvented: the vocabulary is whatever the legacy table
    already carries, which is what ADR 0005 means by consuming `plotting_rules`
    unchanged. Columns are kept only if they are usable as a color -- numeric,
    or categorical with few enough levels to be a legend rather than a smear.
    """
    if not os.path.exists(path):
        return {}
    import pandas as pd

    frame = pd.read_csv(path, sep="\t")
    if "protid" not in frame.columns:
        return {}
    frame = frame.set_index("protid").reindex(protids)
    overlays = {}
    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            overlays[column] = {
                "kind": "continuous",
                "values": [None if pd.isna(v) else float(v) for v in values],
            }
        else:
            levels = values.dropna().astype(str).unique()
            if 1 < len(levels) <= 24:
                overlays[column] = {
                    "kind": "categorical",
                    "values": [None if pd.isna(v) else str(v) for v in values],
                }
    return overlays


#: A feature block may expose its columns as overlays only if it has few enough
#: of them to name. The biophysical block has four; the 3Di block has 4,982
#: 3-mer frequencies, and putting those in a dropdown would be absurd rather
#: than merely long. 16 is chosen to be comfortably above the widest plausible
#: descriptor set and far below any k-mer vocabulary -- there is no continuum
#: between the two, so the exact value is not load-bearing.
MAX_NAMED_BLOCK_COLUMNS = 16


def _column_shares(output_dir: str, config, space) -> list:
    """Each named column's share of a single-block space's squared distance.

    The within-block twin of `_contributions`, which apportions a FUSED space
    between its blocks. This apportions an unfused feature space between its
    own columns, and it is the same question one level down: what is this
    picture actually made of.

    The arithmetic is exact rather than an analogy. Euclidean distance on raw
    columns is a sum of per-column squared differences, so a column's share of
    the total variance IS its share of the squared distance, averaged over
    pairs. Nothing here is a proxy for the thing; it is the thing.

    Returns ``[]`` unless the space has exactly one block and that block names
    its columns -- with several blocks the apportionment is the fused
    contribution, which the run already computes, and with 4,982 unnamed 3-mer
    columns there is nothing a reader could do with the answer.
    """
    import numpy as np

    blocks = list(getattr(space, "blocks", ()) or ())
    if len(blocks) != 1:
        return []
    block_id = blocks[0]
    directory = os.path.join(output_dir, layout.BLOCKS_DIRNAME, block_id)
    manifest = _read_json(os.path.join(directory, layout.BLOCK_MANIFEST_FILENAME)) or {}
    names = list((manifest.get("extra") or {}).get("descriptors") or [])
    features_path = os.path.join(directory, "features.npy")
    if not names or len(names) > MAX_NAMED_BLOCK_COLUMNS or not os.path.exists(features_path):
        return []
    values = np.load(features_path)
    if values.ndim != 2 or values.shape[1] != len(names):
        return []
    variance = values.var(axis=0)
    total = float(variance.sum())
    if total <= 0:
        return []
    # What the block's own declared normalization would have produced. Every
    # column standardized has equal variance by construction, so this is 1/n --
    # stated as a number rather than left for the reader to work out, because
    # the gap between the two columns is the whole point of showing either.
    declared = getattr(config.blocks.get(block_id), "normalization", None)
    return [
        {
            "column": name,
            "variance": float(variance[index]),
            "share": float(variance[index] / total),
            "share_if_standardized": 1.0 / len(names),
            "declared_normalization": declared,
        }
        for index, name in enumerate(names)
    ]


def _block_column_overlays(output_dir: str, config, protids: list) -> dict:
    """Per-descriptor overlays, read from the blocks that a geometry is built ON.

    THIS IS THE ONLY OVERLAY SOURCE THAT COLOURS A MAP BY ITS OWN INPUT. Every
    other overlay comes from the aggregated features table and is, by
    construction, something the geometry never saw. These are the columns the
    `physicochemistry` space is a reduction OF, so colouring that map by one of
    them answers "which descriptor is this picture actually made of" -- a
    question no feature-table overlay can answer.

    That is worth having for a specific reason. The biophysical block declares
    `normalization="zscore_within"` and nothing reads the field (FOLLOWUPS #32),
    so its four columns enter the distance in their raw units: isoelectric point
    runs about 4 to 12 and charge per residue about -0.1 to 0.1. The prediction
    is that the map is mostly pI. Before this, that was an argument; now it is
    something a reader can see by changing a dropdown.

    Named ``<block_id>:<column>`` so the source travels with the number and a
    descriptor cannot collide with a same-named column of the features table.

    Only blocks whose manifest names its columns are read, and only when there
    are few enough of them -- see `MAX_NAMED_BLOCK_COLUMNS`.
    """
    import numpy as np

    overlays: dict = {}
    order = {protid: i for i, protid in enumerate(protids)}
    for block_id in sorted(getattr(config, "blocks", {}) or {}):
        directory = os.path.join(output_dir, layout.BLOCKS_DIRNAME, block_id)
        manifest = _read_json(os.path.join(directory, layout.BLOCK_MANIFEST_FILENAME)) or {}
        names = list((manifest.get("extra") or {}).get("descriptors") or [])
        if not names or len(names) > MAX_NAMED_BLOCK_COLUMNS:
            continue
        features_path = os.path.join(directory, "features.npy")
        protids_path = os.path.join(directory, "protids.txt")
        if not (os.path.exists(features_path) and os.path.exists(protids_path)):
            continue
        with open(protids_path) as handle:
            block_protids = [line.strip() for line in handle if line.strip()]
        values = np.load(features_path)
        # The manifest names the columns and the array supplies them; if the two
        # disagree the block was written by a different version of the provider
        # and guessing which columns are which would mislabel every point.
        if values.ndim != 2 or values.shape[1] != len(names):
            continue
        if values.shape[0] != len(block_protids):
            continue
        for index, name in enumerate(names):
            column = [None] * len(protids)
            for row, protid in enumerate(block_protids):
                at = order.get(protid)
                if at is not None:
                    value = float(values[row, index])
                    column[at] = None if not np.isfinite(value) else value
            overlays[f"{block_id}:{name}"] = {"kind": "continuous", "values": column}
    return overlays


def build_payload(config, output_dir: str, analysis_name: str = "analysis") -> ExplorerPayload:
    """Read a finished run and assemble the explorer's data.

    Only spaces that produced an embedding are included: a space that was
    skipped because its provider was unavailable has nothing to draw, and
    inventing an empty panel for it would suggest the run covered more than it
    did (ADR 0006 -- a missing optional dependency is a reduced result).
    """
    spaces, index_order = [], None
    for space_id, space in sorted(config.spaces.items()):
        directory = os.path.join(output_dir, "spaces", space_id)
        embeddings, protids = {}, None
        for reducer in space.reducers:
            path = os.path.join(directory, layout.embedding_filename(reducer))
            if not os.path.exists(path):
                continue
            protids, coordinates = read_embedding(path)
            embeddings[reducer] = coordinates
        if not embeddings:
            continue
        diagnostics = _read_json(os.path.join(directory, layout.DIAGNOSTICS_FILENAME)) or {}
        summary = {k: diagnostics[k] for k in SUMMARY_SECTIONS if k in diagnostics}
        # The same shape as the faithfulness trim below: keep the section, drop
        # the one key inside it that is a table rather than a summary.
        if summary.get("censoring"):
            summary["censoring"] = [
                {k: v for k, v in entry.items() if k not in CENSORING_DROPPED_KEYS}
                for entry in summary["censoring"]
            ]
        summary["stability"] = diagnostics.get("stability")
        summary["faithfulness"] = [
            {
                k: v
                for k, v in entry.items()
                if k not in ("trustworthiness", "continuity", "protids")
            }
            for entry in diagnostics.get("faithfulness") or []
        ]
        contributions = _contributions(directory, embeddings)
        axes = {reducer: _reducer_axes(directory, reducer) for reducer in embeddings}
        axes = {reducer: facts for reducer, facts in axes.items() if facts}
        spaces.append(
            SpacePayload(
                space_id=space_id,
                protids=protids,
                embeddings=embeddings,
                clusters=_read_clusters(os.path.join(directory, layout.CLUSTERS_FILENAME)),
                readable=_readable_mask(directory, diagnostics, protids),
                verdict=space_verdict(diagnostics, len(protids)),
                diagnostics=summary,
                contributions=contributions,
                stability=_stability_series(directory, protids),
                column_shares=_column_shares(output_dir, config, space),
                axes=axes,
                description=descriptions.describe_space(
                    space_id,
                    strategy=getattr(space, "strategy", "none"),
                    blocks=_space_blocks(output_dir, config, space),
                    contributions=contributions,
                    axes=axes,
                ),
            )
        )
        index_order = index_order or protids

    comparisons = _read_comparisons(layout.summary_path(output_dir))
    overlays = _overlays_from_features(
        _features_table(output_dir, analysis_name), index_order or []
    )
    # The blocks' own columns, added after the feature table so a descriptor
    # never silently displaces a same-named column the table already had.
    for name, overlay in _block_column_overlays(output_dir, config, index_order or []).items():
        overlays.setdefault(name, overlay)
    provenance = _provenance(output_dir, config, spaces)
    records = _records(output_dir, index_order or [])
    pipeline = _pipeline(config, spaces)
    structural = _structural_space(config, spaces)
    tm_matrix = _tm_matrix(config, spaces, structural)
    censoring = _censoring(config, spaces, structural)
    comparison = _censoring_comparison(output_dir, config, spaces)
    if comparison:
        censoring = dict(censoring or {})
        censoring["comparison"] = comparison
    # What the page HAS, named the way the catalogue names it. A panel is
    # drawable when everything it needs is in here; anything absent becomes the
    # panel's printed "awaiting ..." line rather than a blank.
    available = set()
    if comparisons:
        available.add("comparisons")
    if any(space.contributions for space in spaces):
        available.add("fused_spaces")
    # The presence of overlays used to stand in for this. It is a different
    # file: overlays come from the aggregated features table and records from
    # `uniprot_features.tsv`, so a run with one and not the other advertised a
    # records panel it could not fill.
    if records:
        available.add("records")
    # The overlay flavour is the one the source calls geometry-preserving, and
    # whether this run has any is a fact about the run: a cohort with no feature
    # table has no overlay to keep always on. Named here rather than inferred in
    # the page, like every other entry in this set.
    if overlays:
        available.add("overlays")
    if pipeline:
        available.add("pipeline")
    if tm_matrix:
        available.add("matrix")
    if censoring:
        available.add("censoring")
    if any(space.stability for space in spaces):
        available.add("stability_series")
    return ExplorerPayload(
        analysis_name=analysis_name,
        spaces=spaces,
        comparisons=comparisons,
        overlays=overlays,
        provenance=provenance,
        panels=panels.catalogue_for(available),
        sheets=panels.sheet_titles(),
        hover=_hover_fields(output_dir, {p for space in spaces for p in space.protids}),
        records=records,
        pipeline=pipeline,
        tm_matrix=tm_matrix,
        censoring=censoring,
        thresholds=_thresholds(),
    )


def _features_table(output_dir: str, analysis_name: str) -> str:
    """Where `aggregate_features` actually put the table.

    It is `final_results/{analysis_name}_aggregated_features.tsv`, not
    `aggregated_features.tsv` at the top of the output directory -- the first
    draft guessed the latter and silently produced an explorer with no overlays
    at all, which looks exactly like a run whose features table was empty.
    The analysis name is a config value, so the name is constructed rather than
    globbed; the glob is the fallback for a run whose name is not to hand.
    """
    import glob

    named = os.path.join(
        output_dir, "final_results", layout.aggregated_features_filename(analysis_name)
    )
    if os.path.exists(named):
        return named
    matches = sorted(
        glob.glob(os.path.join(output_dir, "final_results", "*_aggregated_features.tsv"))
    )
    return matches[0] if matches else named


def _read_comparisons(path: str) -> list:
    """The pairwise summary, each row carrying its own per-protein detail.

    `summary.tsv` is aggregate only -- one row per pair, with `jaccard_mean` and
    friends. The per-protein neighbourhood Jaccard lives beside it in
    `{space_a}__vs__{space_b}.tsv`, and the explorer's disagreement mode needs
    exactly that: it averages each protein's Jaccard across every pair it
    appears in.

    Attaching it here rather than in the template is what makes the feature
    work at all. Before this the template looped over these rows testing
    `if (!row.per_protein) continue`, no row had ever carried the key, the map
    it built was empty, and every protein coloured `null` -- so ADR 0005 item
    4's headline "one click, not buried in a menu" toggled a button that
    conveyed nothing. Found by opening the page, which is the only thing that
    could find it.
    """
    if not os.path.exists(path):
        return []
    import pandas as pd

    frame = pd.read_csv(path, sep="\t")
    directory = os.path.dirname(path)
    rows = []
    for record in frame.to_dict("records"):
        row = {k: (None if pd.isna(v) else v) for k, v in record.items()}
        row["per_protein"] = _per_protein_jaccard(directory, row.get("space_a"), row.get("space_b"))
        rows.append(row)
    return rows


def _per_protein_jaccard(directory: str, space_a, space_b) -> dict:
    """`{protid: neighborhood_jaccard}` for one compared pair, or `{}`.

    NaN is dropped rather than carried as null: a protein whose Jaccard is
    undefined for this pair should not drag its mean down, and the template
    already treats an absent protid as "no disagreement value" rather than as
    zero. Averaging a missing comparison in as 0.0 would report maximal
    disagreement for a pair that was never measured.
    """
    if not space_a or not space_b:
        return {}
    import pandas as pd

    path = os.path.join(directory, layout.pair_filename(space_a, space_b))
    if not os.path.exists(path):
        return {}
    frame = pd.read_csv(path, sep="\t")
    if "protid" not in frame or "neighborhood_jaccard" not in frame:
        return {}
    pairs = zip(frame["protid"], frame["neighborhood_jaccard"])
    return {str(protid): float(value) for protid, value in pairs if not pd.isna(value)}


def _space_manifest(directory: str, reducers) -> dict:
    """A space's manifest, under whichever reducer wrote one.

    `reduce_space` writes one manifest per reducer (`manifest_pca_umap.json`)
    and `diagnose_space` writes one for the diagnostics run. Neither writes a
    bare `manifest.json` -- that name belongs to the *block* store, which is a
    different directory. Reading it here is GE.5. Any of these carries
    the provenance this needs; the reducers are tried in the order the panel
    would draw them so the reported `params` match what is on screen.
    """
    for reducer in list(reducers) + [layout.DIAGNOSTICS_MANIFEST_KEY]:
        manifest = _read_json(os.path.join(directory, layout.manifest_filename(reducer)))
        if manifest:
            return manifest
    return {}


def _reducer_axes(directory: str, reducer: str) -> dict:
    """What ONE reducer actually did to this space, read from its own manifest.

    The axes paragraph used to be a typed sentence saying "UMAP1 and UMAP2 of a
    30-component PCA" under every map. Thirty is what the config REQUESTS; PCA
    returns min(n_components, n_samples, n_features), and a four-column
    biophysics block can only yield four. So the sentence was wrong on
    physicochemistry on both shipped cohorts -- sixteen times on the built page
    -- while the correct number sat in `n_components_used` in the manifest.

    Returns {} when the manifest or its steps are absent, so a run that predates
    the steps key says nothing rather than guessing.
    """
    manifest = _read_json(os.path.join(directory, layout.manifest_filename(reducer))) or {}
    steps = (manifest.get("extra") or {}).get("steps") or []
    if not steps:
        return {}
    pca = next((step for step in steps if step.get("reducer") == "pca"), {})
    return {
        "axis_names": list(steps[-1].get("column_names") or []),
        "pca_components_requested": pca.get("n_components_requested"),
        "pca_components_used": pca.get("n_components_used"),
    }


def _contributions(directory: str, reducers) -> list:
    """What each block actually contributed to a fused space.

    ADR 0002 requires this to be computed, recorded, logged *and* displayed --
    "no fused map renders without it visible". It was the first three only. The
    nominal and realized shares are both carried because they differ: `early`
    concatenates features, so a 50/50 request over blocks of 11 and 4 columns
    realizes 73/27, and showing only the request would misreport the map.
    """
    fusion = _space_manifest(directory, reducers).get("extra", {}).get("fusion", {})
    entries = fusion.get("contributions", [])
    # One block contributes all of itself. Rendering "tmscore 100%" on every
    # unfused panel is the noise half of the rule that a diagnostic which always
    # fires says nothing, and it is what the docstring above promises not to do.
    if len(entries) < 2:
        return []
    return [
        {
            "block_id": entry.get("block_id"),
            "weight": entry.get("weight"),
            "share": entry.get("share"),
            "realized_share": entry.get("realized_share"),
        }
        for entry in entries
    ]


def _pipeline(config, spaces: list) -> dict:
    """The resolved pipeline as boxes and arrows, for THIS run.

    The source's 2.01 asks for a network diagram -- each box a tensor, each
    arrow an operation -- and names two things that surprise most readers. Both
    are facts about this run rather than about the general case, so both are
    read out of the resolved config and the written diagnostics instead of being
    drawn as prose:

    * **the input to PCA is the similarity matrix itself**, for any block whose
      representation is `profile`: a protein's feature vector is its own row of
      the matrix, so the matrix is the tensor and not a step towards one.
    * **clustering runs on a different graph than the map**. Leiden builds its
      own kNN graph, and `partition.n_neighbors` in the diagnostics is that
      graph's k while the reducer's `n_neighbors` is the map's. On the shipped
      cohorts these are 37 and 15, which is the whole point of saying it.

    Returns `{}` when there is nothing to draw, so the panel says it is
    awaiting an input rather than rendering an empty diagram.
    """
    blocks = []
    for block_id in sorted(getattr(config, "blocks", {}) or {}):
        block = config.blocks[block_id]
        representation = getattr(block, "representation", None)
        blocks.append(
            {
                "block_id": block_id,
                "provider": getattr(block, "provider", "") or "",
                "representation": representation or "features",
                # The one that surprises people. Said per block, because it is
                # true of `profile` blocks and false of the others.
                "is_profile": representation == "profile",
                "metric": getattr(block, "metric", "") or "",
            }
        )
    rows = []
    for space in spaces:
        space_config = (getattr(config, "spaces", {}) or {}).get(space.space_id)
        reducer_params = dict(getattr(space_config, "reducer_params", {}) or {})
        partition = (space.diagnostics or {}).get("partition") or {}
        for reducer in sorted(space.embeddings):
            params = dict(reducer_params.get(reducer) or {})
            rows.append(
                {
                    "space_id": space.space_id,
                    "blocks": list(getattr(space_config, "blocks", ()) or ()),
                    "strategy": getattr(space_config, "strategy", "none") or "none",
                    "reducer": reducer,
                    # None, not a number, when the config named none: the
                    # reducer's own default is not the same fact as a value
                    # somebody chose, and printing the default here would claim
                    # the config said something it did not.
                    "map_n_neighbors": params.get("n_neighbors"),
                    "n_pcs": partition.get("n_pcs"),
                    "cluster_n_neighbors": partition.get("n_neighbors"),
                    "resolution": partition.get("resolution"),
                    "n_clusters": partition.get("n_clusters"),
                    "cluster_source": partition.get("source"),
                }
            )
    if not blocks and not rows:
        return {}
    return {"blocks": blocks, "rows": rows}


def _thresholds() -> dict:
    """The judgement lines the page draws, read from the code that enforces them.

    Retyping 0.30 into the template would let the picture and the verdict
    disagree the day either moves -- the same reasoning as the signal
    inventory, which reads its rows out of the guard rather than restating
    them.
    """
    from diagnostics.embedding import DISTORTED_THRESHOLD
    from diagnostics.stability import COIN_FLIP_THRESHOLD

    return {"coin_flip": float(COIN_FLIP_THRESHOLD), "distorted": float(DISTORTED_THRESHOLD)}


def _matrix_path_for(config, space_id: str) -> str:
    """The matrix behind ONE named space, by that space's own block.

    Deliberately not "the first block whose provider is tmscore". A run can
    carry more than one -- the censoring comparison needs a capped twin
    alongside the shipped block -- and a scan over `config.blocks` would return
    whichever the mapping happened to yield first. That is dict insertion order,
    which is a property of how the config was written rather than of what the
    panel is describing, and it would let the matrix panel silently draw one
    cohort's matrix while its caption described another.
    """
    space = (getattr(config, "spaces", {}) or {}).get(space_id)
    blocks = getattr(config, "blocks", {}) or {}
    for block_id in getattr(space, "blocks", ()) or ():
        block = blocks.get(block_id)
        if getattr(block, "provider", "") == "tmscore":
            return (getattr(block, "params", {}) or {}).get("matrix_path") or ""
    return ""


def _structural_space(config, spaces: list) -> str:
    """Which space the matrix and censoring panels describe, resolved from data.

    Both panels used to default to the literal id ``"structure"``. That is a
    naming convention, not a fact about a run: a cohort whose structural space
    is called ``shape`` got an empty matrix panel and a censoring panel that
    said its input did not exist, while the matrix sat on disk. Nothing failed,
    because an absent payload key and an unbuildable panel look identical.

    THE RULE, and the ordering in it is deliberate: the FIRST space in config
    order carrying a block whose provider is ``tmscore`` and whose
    ``matrix_path`` exists on disk. First-in-config-order is what keeps actin_B
    on ``structure`` rather than on the capped twin ``structure_capped`` that
    the censoring comparison adds -- both are tmscore blocks with real
    matrices, so any rule that did not fix an order would pick between them by
    dict insertion order, which is a property of how the config was typed.

    Returns "" when no space qualifies, which makes both panels refuse with
    their own reason rather than draw someone else's matrix.
    """
    for space_id in getattr(config, "spaces", {}) or {}:
        space = next((s for s in spaces if s.space_id == space_id), None)
        if space is None:
            continue
        path = _matrix_path_for(config, space_id)
        if path and os.path.exists(path):
            return space_id
    return ""


#: How many quantisation levels the heatmap ships. uint8 over the observed
#: range: at n=367 the full matrix is 135 KB raw and about 180 KB in base64,
#: against 1.1 MB for float64. See `_tm_matrix` for why the FULL matrix and not
#: a triangle.
_HEATMAP_LEVELS = 255


def _tm_matrix(config, spaces: list, sort_space: str = "") -> dict:
    """The cohort's similarity matrix, cluster-sorted and quantised for display.

    THE WHOLE MATRIX, NOT A TRIANGLE, and that is a measurement rather than a
    preference. These matrices are **not symmetric**: on the two shipped cohorts
    19-20 % of pairs have ``a_ij != a_ji``, about 1 % differ by more than 0.05,
    and the largest disagreement is 0.66. Shipping one triangle would silently
    choose one of two real values for one pair in five, and the pairs where the
    choice matters most are exactly the interesting ones. The extra bytes buy
    the reader the ability to see the asymmetry, so the measured asymmetry
    travels with it.

    QUANTISED TO 255 LEVELS, WHICH IS A LOSS AND IS DECLARED. The largest
    resulting error is about 0.0018, which is coarser than the four significant
    figures foldseek emits. That is invisible in a heat map and fatal in a
    number, so `max_error` rides in the payload and the panel prints it: a cell
    here is a colour, not a measurement.

    Sorted by the clusters of ``sort_space`` and, within a cluster, by protid,
    so the order is deterministic and two runs of the same inputs produce the
    same bytes -- the same property the provenance footer refuses a timestamp
    for.

    Returns ``{}`` when there is no matrix to read or nothing to sort by, so the
    panel keeps naming the input it wants instead of drawing an empty square.
    """
    import base64

    import numpy as np

    space = next((s for s in spaces if s.space_id == sort_space), None)
    if space is None:
        return {}
    path = _matrix_path_for(config, sort_space)
    if not path or not os.path.exists(path):
        return {}

    from matrix_io import load_labeled_matrix

    matrix = load_labeled_matrix(path, repair=True)
    # Every reorder below goes through the labelled frame. ADR 0007: this data
    # is never indexed positionally, and a cluster sort is precisely the
    # operation that would silently scramble it.
    frame = matrix.to_frame()
    frame.index = [str(label) for label in frame.index]
    frame.columns = [str(label) for label in frame.columns]
    known = set(frame.index) & set(frame.columns)
    order = [p for p in space.protids if p in known]
    if not order:
        return {}
    order.sort(key=lambda protid: (space.clusters.get(protid, "~"), protid))
    values = frame.loc[order, order].to_numpy(dtype=float)

    finite = values[np.isfinite(values)]
    if not finite.size:
        return {}
    low, high = float(finite.min()), float(finite.max())
    span = high - low
    if span <= 0:
        return {}
    scaled = np.rint((values - low) / span * _HEATMAP_LEVELS)
    quantised = np.clip(scaled, 0, _HEATMAP_LEVELS).astype(np.uint8)
    restored = quantised.astype(float) / _HEATMAP_LEVELS * span + low
    max_error = float(np.nanmax(np.abs(restored - values)))

    upper = np.triu_indices(len(order), 1)
    gaps = np.abs(values - values.T)[upper]
    asymmetry = {
        "n_pairs": int(gaps.size),
        "n_asymmetric": int((gaps > 1e-9).sum()),
        "max_gap": float(gaps.max()) if gaps.size else 0.0,
        "n_above_0_05": int((gaps > 0.05).sum()),
    }

    # Run-length bands rather than a label per row: at n=367 a per-row list is
    # 367 strings for seven bands, and the panel draws bands.
    bands: list = []
    for protid in order:
        cluster = space.clusters.get(protid, "")
        if bands and bands[-1]["cluster"] == cluster:
            bands[-1]["count"] += 1
        else:
            bands.append({"cluster": cluster, "count": 1})

    return {
        "protids": order,
        "n": len(order),
        "values": base64.b64encode(quantised.tobytes()).decode("ascii"),
        "low": low,
        "high": high,
        "levels": _HEATMAP_LEVELS,
        "max_error": max_error,
        "bands": bands,
        "sorted_by": sort_space,
        "censoring_rate": float(matrix.censoring_rate),
        "asymmetry": asymmetry,
        # Named here, once, so no renderer has to remember it. FOLLOWUPS #60:
        # the score is a foldseek 3Di+AA alignment score and not TM-align's.
        "value_label": "3Di+AA-derived TM (FOLLOWUPS #60), not TM-align output",
    }


def _censoring(config, spaces: list, sort_space: str = "") -> dict:
    """What the per-query cap removed from this cohort's own matrix.

    `matrix_io.summarize_censoring` already computes all of this and is what the
    tmscore block's manifest records, so this reads the guard rather than
    recomputing it -- the same reasoning as the signal inventory.

    THE ROW/COLUMN ASYMMETRY IS THE PART WORTH DRAWING. Foldseek's `--max-seqs`
    bounds how many partners each *query* reports, so a per-query cap piles the
    rows up at one number while the columns stay free. A matrix with the same
    censoring rate spread evenly is a different thing entirely, and
    `cap_detected` refuses to call it a cap unless both halves hold.

    A cohort whose matrix has no censored cells returns its zero rather than
    nothing. That is a real and otherwise invisible property -- both shipped
    cohorts were built from exhaustive matrices, and a reader who assumed the
    default capped pipeline would misread every panel on the page.
    """
    import numpy as np

    space = next((s for s in spaces if s.space_id == sort_space), None)
    if space is None:
        return {}
    path = _matrix_path_for(config, sort_space)
    if not path or not os.path.exists(path):
        return {}

    from matrix_io import load_labeled_matrix, summarize_censoring

    matrix = load_labeled_matrix(path, repair=True)
    summary = summarize_censoring(matrix)
    per_protid = matrix.censoring_rate_per_protid()
    rates = [
        {"protid": str(protid), "rate": float(rate)}
        for protid, rate in per_protid.items()
        if str(protid) in set(space.protids)
    ]
    rates.sort(key=lambda row: (-row["rate"], row["protid"]))
    values = np.array([row["rate"] for row in rates], dtype=float)
    return {
        "summary": summary,
        # The whole per-protein list, so the panel can draw a distribution
        # rather than three summary numbers. At n=367 this is small.
        "rates": rates,
        "quantiles": {
            "min": float(values.min()) if values.size else 0.0,
            "median": float(np.median(values)) if values.size else 0.0,
            "max": float(values.max()) if values.size else 0.0,
        },
        "n_proteins": len(rates),
    }


#: Above this Procrustes disparity two 2-D layouts share so little common shape
#: that superimposing them is meaningless, so the explorer refuses to draw a
#: protein's two positions as a displacement.
#:
#: 0.5 is not a taste. Measured on actin_B: reducing the SAME matrix with the
#: same parameters and only a different seed gives 0.010-0.031, so the reducer
#: itself is stable to three decimal places here; two genuinely different
#: modalities (structure against local_structure) give 0.862. A pair above 0.5
#: is therefore nowhere near reducer noise and already past "as different as two
#: different modalities", which is the point where a connecting line stops
#: describing movement and starts inventing it.
SUPERIMPOSABLE_THRESHOLD = 0.5


def _censoring_comparison(output_dir: str, config, spaces: list) -> dict:
    """What censoring did to a map, measured against its uncensored twin.

    `diagnostics.censoring_comparison` names two spaces built from the same
    proteins whose only difference is which pairs were measured. Everything here
    comes from the coregistration the pipeline already ran on that pair.

    THE DRAG LINE IS THE OBVIOUS PANEL AND IT IS THE WRONG ONE. Drawing each
    protein's censored position joined to its uncensored one reads as "this is
    how far the cap moved it", and that reading needs the two layouts to share a
    frame. They do not: on actin_B the pair's Procrustes disparity is 0.968,
    which is worse than structure against an entirely different modality. So the
    displacement is refused, with the measured disparity as the reason rather
    than an argument, and what is drawn instead is neighbourhood retention --
    which is invariant to rotation, reflection and scale and therefore says
    something true whatever the frames do.
    """
    # A direct attribute access, not `getattr(..., default)`. The field always
    # exists on a validated config, and `test_diagnostics_config` reads the AST
    # to prove every diagnostics field is consumed somewhere -- it cannot see a
    # getattr, so writing one here would have made this field look dead to the
    # guard that exists to catch exactly the #29/#32 defect shape.
    pair = tuple(config.diagnostics.censoring_comparison or ())
    if len(pair) != 2:
        return {}
    censored_id, reference_id = pair
    known = {space.space_id for space in spaces}
    if censored_id not in known or reference_id not in known:
        return {}

    directory = os.path.join(output_dir, "coregistration")
    per_protein = _per_protein_jaccard(directory, censored_id, reference_id)
    if not per_protein:
        per_protein = _per_protein_jaccard(directory, reference_id, censored_id)
    summary = {}
    for row in _read_comparisons(layout.summary_path(output_dir)):
        names = {row.get("space_a"), row.get("space_b")}
        if names == {censored_id, reference_id}:
            summary = row
            break
    if not summary:
        return {}

    disparity = summary.get("procrustes_disparity")
    retained = sorted(
        ({"protid": protid, "jaccard": value} for protid, value in per_protein.items()),
        key=lambda row: (row["jaccard"], row["protid"]),
    )
    # Every OTHER pair this run compared, so the reader can see what the number
    # means without being told. A disparity is only interpretable next to the
    # disparities of pairs whose difference is understood.
    context = [
        {
            "space_a": row.get("space_a"),
            "space_b": row.get("space_b"),
            "jaccard_mean": row.get("jaccard_mean"),
            "procrustes_disparity": row.get("procrustes_disparity"),
        }
        for row in _read_comparisons(layout.summary_path(output_dir))
        if {row.get("space_a"), row.get("space_b")} != {censored_id, reference_id}
    ]
    superimposable = disparity is not None and disparity <= SUPERIMPOSABLE_THRESHOLD
    return {
        "censored_space": censored_id,
        "reference_space": reference_id,
        "jaccard_mean": summary.get("jaccard_mean"),
        "procrustes_disparity": disparity,
        "cluster_ari": summary.get("cluster_ari"),
        "k": summary.get("k"),
        "retained": retained,
        "context": context,
        "threshold": SUPERIMPOSABLE_THRESHOLD,
        "superimposable": bool(superimposable),
    }


def _provenance(output_dir: str, config, spaces: list) -> dict:
    """The footer. Always visible, never collapsed (ADR 0005 item 8).

    Deliberately carries no timestamp. Two runs of the same inputs must produce
    the same file, and a generation time would be the one field guaranteeing
    they never do -- the same reasoning that makes the pipeline's outputs
    byte-comparable in the first place.
    """
    manifests = {}
    for space in spaces:
        # `manifest_{reducer}.json`, not `manifest.json`. The store has never
        # written the latter, so this block silently produced `{}` and the
        # provenance footer rendered an empty list -- visible only by opening
        # the file and looking, which is how it was found.
        manifest = _space_manifest(
            os.path.join(output_dir, "spaces", space.space_id), space.embeddings
        )
        if manifest:
            manifests[space.space_id] = {
                "cache_key": manifest.get("cache_key"),
                "provider": manifest.get("provider"),
                "params": manifest.get("params"),
                "versions": manifest.get("versions"),
            }
    return {
        "n_spaces": len(spaces),
        "n_proteins": len(spaces[0].protids) if spaces else 0,
        "cohort_rule": getattr(config.cohort, "selection", None),
        "diagnostics_k": config.diagnostics.k,
        "manifests": manifests,
    }
