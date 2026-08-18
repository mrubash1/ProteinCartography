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
SUMMARY_SECTIONS = ("redundancy", "negative_controls", "resolution_sweep", "partition")


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

    def to_dict(self) -> dict:
        return {
            "space_id": self.space_id,
            "protids": self.protids,
            "embeddings": self.embeddings,
            "clusters": self.clusters,
            "readable": self.readable,
            "verdict": self.verdict,
            "diagnostics": self.diagnostics,
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
            f"mean neighbourhood stability is {stability['stability_mean']:.2f}, at or "
            "below the coin-flip level"
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
        path = os.path.join(directory, f"faithfulness_{entry.get('reducer')}.tsv")
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


@dataclass(frozen=True)
class ExplorerPayload:
    """Everything the single HTML file embeds."""

    analysis_name: str
    spaces: list
    comparisons: list
    overlays: dict
    provenance: dict

    def to_dict(self) -> dict:
        return {
            "analysis_name": self.analysis_name,
            "spaces": [space.to_dict() for space in self.spaces],
            "comparisons": self.comparisons,
            "overlays": self.overlays,
            "provenance": self.provenance,
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
            path = os.path.join(directory, f"embedding_{reducer}.tsv")
            if not os.path.exists(path):
                continue
            protids, coordinates = read_embedding(path)
            embeddings[reducer] = coordinates
        if not embeddings:
            continue
        diagnostics = _read_json(os.path.join(directory, "diagnostics.json")) or {}
        summary = {k: diagnostics[k] for k in SUMMARY_SECTIONS if k in diagnostics}
        summary["stability"] = diagnostics.get("stability")
        summary["faithfulness"] = [
            {
                k: v
                for k, v in entry.items()
                if k not in ("trustworthiness", "continuity", "protids")
            }
            for entry in diagnostics.get("faithfulness") or []
        ]
        spaces.append(
            SpacePayload(
                space_id=space_id,
                protids=protids,
                embeddings=embeddings,
                clusters=_read_clusters(os.path.join(directory, "clusters.tsv")),
                readable=_readable_mask(directory, diagnostics, protids),
                verdict=space_verdict(diagnostics, len(protids)),
                diagnostics=summary,
            )
        )
        index_order = index_order or protids

    comparisons = _read_comparisons(os.path.join(output_dir, "coregistration", "summary.tsv"))
    overlays = _overlays_from_features(
        _features_table(output_dir, analysis_name), index_order or []
    )
    provenance = _provenance(output_dir, config, spaces)
    return ExplorerPayload(
        analysis_name=analysis_name,
        spaces=spaces,
        comparisons=comparisons,
        overlays=overlays,
        provenance=provenance,
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

    named = os.path.join(output_dir, "final_results", f"{analysis_name}_aggregated_features.tsv")
    if os.path.exists(named):
        return named
    matches = sorted(
        glob.glob(os.path.join(output_dir, "final_results", "*_aggregated_features.tsv"))
    )
    return matches[0] if matches else named


def _read_comparisons(path: str) -> list:
    if not os.path.exists(path):
        return []
    import pandas as pd

    frame = pd.read_csv(path, sep="\t")
    rows = []
    for record in frame.to_dict("records"):
        rows.append({k: (None if pd.isna(v) else v) for k, v in record.items()})
    return rows


def _provenance(output_dir: str, config, spaces: list) -> dict:
    """The footer. Always visible, never collapsed (ADR 0005 item 8).

    Deliberately carries no timestamp. Two runs of the same inputs must produce
    the same file, and a generation time would be the one field guaranteeing
    they never do -- the same reasoning that makes the pipeline's outputs
    byte-comparable in the first place.
    """
    manifests = {}
    for space in spaces:
        path = os.path.join(output_dir, "spaces", space.space_id, "manifest.json")
        manifest = _read_json(path)
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
