#!/usr/bin/env python
"""Compare co-registered spaces, and say what the comparison rests on.

The snakemake entry point for :mod:`coregistration`. Reads every space named in
``coregistration.compare`` out of the block store, aligns them all to one
protein index, and writes the cross-space metrics for every pair.

Three files land under ``OUTPUT_DIR/coregistration/``:

``index.json``
    The shared index and how each space reached it, including every protein a
    space measured that the comparison could not use. Read this before the
    numbers: it is what says whether they describe the cohort or an overlap.
``summary.tsv``
    One row per pair -- mean and median neighborhood Jaccard, mean rank
    correlation, Procrustes disparity, and the diagnostics that qualify them.
``{a}__vs__{b}.tsv``
    Per protein, for one pair. This is the file a disagreement mode reads.

Additive, like every other rule in this work: nothing existing consumes any of
it, and the rule stays out of the DAG entirely unless a config asks for a
comparison.
"""

from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from config_io import load_config
from config_schema import from_legacy
from coregistration import GEOMETRY_CAVEATS, CoregistrationError, compare_pair, shared_index
from reduce_space import features_for
from spaces.manifest import Manifest
from spaces.store import BlockStore

COREGISTRATION_SUBDIR = "coregistration"
INDEX_FILENAME = "index.json"
SUMMARY_FILENAME = "summary.tsv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--configfile", required=True)
    parser.add_argument("-o", "--output-dir", required=True, help="the run's output directory")
    parser.add_argument(
        "--embedding",
        action="append",
        default=[],
        metavar="SPACE_ID=PATH",
        help=(
            "a space's 2-D embedding, for the Procrustes comparison. Named rather "
            "than positional for the reason `compute_block --provider-input` is: a "
            "bare list of paths has to be indexed into, and indexing a labeled thing "
            "positionally is the defect ADR 0007 exists to prevent."
        ),
    )
    return parser.parse_args()


def parse_embeddings(pairs) -> dict:
    """``["structure=/p/e.tsv"]`` -> ``{"structure": "/p/e.tsv"}``."""
    embeddings = {}
    for pair in pairs:
        space_id, separator, path = pair.partition("=")
        if not separator or not space_id.strip():
            raise SystemExit(
                f"--embedding expects SPACE_ID=PATH, got {pair!r}. A path containing "
                "an '=' is fine; the split is on the first one."
            )
        embeddings[space_id.strip()] = path
    return embeddings


def pairs_of(space_ids):
    """Every unordered pair, in config order. ``compare`` is a set to compare,
    not a sequence of comparisons, so all of them are wanted."""
    return [(a, b) for i, a in enumerate(space_ids) for b in space_ids[i + 1 :]]


def read_embedding(path: str, protids) -> np.ndarray:
    """A space's embedding, aligned to the shared index.

    Read through the protid column, never positionally: the embedding was
    written over the space's own protid set, which is a superset of the shared
    index whenever any space dropped something.
    """
    frame = pd.read_csv(path, sep="\t", index_col=0)
    missing = [p for p in protids if p not in frame.index]
    if missing:
        raise CoregistrationError(
            f"{path} has no row for {len(missing)} of the shared index's "
            f"{len(protids)} proteins, starting with {missing[:5]}. The embedding "
            "and the block it came from disagree about the cohort."
        )
    return frame.loc[list(protids)].to_numpy(dtype=np.float64)


def main() -> int:
    args = parse_args()
    config = from_legacy(load_config(args.configfile))
    coregistration = config.coregistration
    compared = list(coregistration.compare)

    output_dir = os.path.join(args.output_dir, COREGISTRATION_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    if len(compared) < 2:
        raise SystemExit(
            f"coregistration.compare names {len(compared)} space(s); comparing needs "
            "at least two. Remove the key to skip co-registration entirely -- the "
            "rule does not enter the DAG when nothing asks for it."
        )

    reference = coregistration.reference_space or compared[0]
    if reference not in compared:
        raise SystemExit(
            f"coregistration.reference_space is {reference!r}, which is not among the "
            f"spaces being compared ({compared}). The reference supplies the shared "
            "index's protein order, so it has to be one of them; add it to "
            "`compare` or pick a different reference."
        )

    check_pair_filenames_are_distinct(compared)

    store = BlockStore(args.output_dir)
    blocks, space_protids = {}, {}
    for space_id in compared:
        space = config.spaces[space_id]
        block, index = features_for(space, store, config)
        blocks[space_id] = block
        space_protids[space_id] = list(index)

    report = shared_index(space_protids, reference=reference)
    with open(os.path.join(output_dir, INDEX_FILENAME), "w") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    # Not a debug aid. A comparison conditioned on an overlap nobody chose looks
    # exactly like one that is not, and this line is the only warning a reader
    # who does not open index.json will get.
    print(f"[coregister] {report.describe()}", file=sys.stderr)

    aligned = {
        space_id: report.index.align(
            space_protids[space_id],
            np.asarray(blocks[space_id].features, dtype=np.float64),
            what=f"space {space_id}",
        )
        for space_id in compared
    }

    supplied = parse_embeddings(args.embedding)
    embeddings = {
        space_id: read_embedding(path, report.index.as_list)
        for space_id, path in supplied.items()
        if space_id in compared
    }

    protids = report.index.as_list
    summaries = []
    for space_a, space_b in pairs_of(compared):
        comparison = compare_pair(
            space_a,
            aligned[space_a],
            space_b,
            aligned[space_b],
            protids,
            k=coregistration.k,
            embedding_a=embeddings.get(space_a),
            embedding_b=embeddings.get(space_b),
        )
        per_protein = pd.DataFrame(
            {
                "neighborhood_jaccard": comparison.jaccard,
                "rank_correlation": comparison.spearman,
            },
            index=pd.Index(protids, name="protid"),
        )
        per_protein.to_csv(os.path.join(output_dir, pair_filename(space_a, space_b)), sep="\t")
        summaries.append(comparison.summary())

    summary_frame = pd.DataFrame(
        [
            {
                "space_a": s["space_a"],
                "space_b": s["space_b"],
                "n_proteins": s["n_proteins"],
                "k": s["diagnostics"]["k"],
                "jaccard_mean": s["jaccard_mean"],
                "jaccard_median": s["jaccard_median"],
                "rank_correlation_mean": s["spearman_mean"],
                "procrustes_disparity": s["procrustes_disparity"],
                "boundary_ties_a": s["diagnostics"]["boundary_ties_a"],
                "boundary_ties_b": s["diagnostics"]["boundary_ties_b"],
                "rank_correlation_undefined": s["diagnostics"]["spearman_undefined"],
            }
            for s in summaries
        ]
    )
    summary_frame.to_csv(os.path.join(output_dir, SUMMARY_FILENAME), sep="\t", index=False)

    manifest = Manifest.build(
        "coregistration",
        "coregistration",
        provider="coregister",
        params={"compare": compared, "reference_space": reference, "k": coregistration.k},
        inputs={
            "block:" + blocks[space_id].spec.id: blocks[space_id].manifest.get("cache_key", "")
            for space_id in compared
        },
        protids=protids,
        extra={
            "index": report.to_dict(),
            "pairs": summaries,
            "geometry_caveats": list(GEOMETRY_CAVEATS),
            "procrustes_spaces": sorted(embeddings),
        },
    )
    manifest.write(os.path.join(output_dir, "manifest.json"))

    print(
        f"[coregister] compared {len(summaries)} pair(s) over {len(protids)} proteins "
        f"-> {output_dir}"
    )
    return 0


#: Separates the two space ids in a per-pair filename.
PAIR_SEPARATOR = "__vs__"


def pair_filename(space_a: str, space_b: str) -> str:
    """The per-protein file for one pair."""
    return f"{space_a}{PAIR_SEPARATOR}{space_b}.tsv"


def check_pair_filenames_are_distinct(space_ids) -> None:
    """Refuse ids that would make two different pairs share a filename.

    Nothing constrains a space id today, so `a` vs `b__vs__c` and `a__vs__b` vs
    `c` both want `a__vs__b__vs__c.tsv`. One would overwrite the other and the
    summary would still list both rows, so the loss would be invisible in every
    file a reader opens.
    """
    offenders = [space_id for space_id in space_ids if PAIR_SEPARATOR in space_id]
    if offenders:
        raise SystemExit(
            f"space id(s) {offenders} contain {PAIR_SEPARATOR!r}, which separates the "
            "two ids in a per-pair filename. Two different pairs could then write to "
            "the same file, with one silently overwriting the other. Rename them."
        )


if __name__ == "__main__":
    raise SystemExit(main())
