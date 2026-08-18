#!/usr/bin/env python
"""Say what a space's map can and cannot be read for.

The snakemake entry point for :mod:`diagnostics`. Every other rule in this work
produces a map or a comparison; this one produces the caveats, and writes them
next to the thing they qualify rather than into a log nobody keeps.

Four questions, in the order a reader should ask them:

**Was the input measured?** ``diagnostics/censoring.py``, already built in group
4 and wired in here. A zero in the similarity matrix means Foldseek did not
report the pair within its per-query cap, not that the proteins are dissimilar
(ADR 0009), and 60.5% of a production matrix is that fill. The part worth
reading is cross-cluster edge retention: weak between-cluster pairs fall off the
cap first, so at high censoring the clusters look crisper while their
arrangement decays into noise.

**Was the cohort fair?** ``cohort_report.json``, written by ``download_pdbs``,
is copied into the space's report rather than recomputed. It already carries
candidates before and after truncation, the selection rule, and the taxonomic
composition of retained against discarded proteins.

**Do the blocks say different things?** ``diagnostics/redundancy.py``. Only for
a space with more than one block, and the answer is the one a contribution
share cannot give.

**Did the map survive two dimensions?** ``diagnostics/embedding.py``, per
reducer and per protein.

Additive, like every rule in this work. Nothing consumes what it writes, and it
stays out of the DAG entirely unless a config asks for a space.
"""

from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
from config_io import load_config
from config_schema import from_legacy
from coregistration import pairwise_distances
from diagnostics.embedding import faithfulness
from diagnostics.redundancy import RedundancyError, redundancy
from reduce_space import features_for
from spaces.manifest import Manifest
from spaces.store import BlockStore

DIAGNOSTICS_FILENAME = "diagnostics.json"

#: The four questions, in the order the module docstring asks them. A section is
#: absent when this run could not answer it -- a single-block space has no
#: redundancy, a cluster-mode run has no cohort report -- so the set that landed
#: is itself information and is recorded in the manifest.
SECTIONS = ("censoring", "cohort", "redundancy", "faithfulness")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--configfile", required=True)
    parser.add_argument("-s", "--space-id", required=True)
    parser.add_argument("-o", "--output-dir", required=True, help="the run's output directory")
    parser.add_argument(
        "--embedding",
        action="append",
        default=[],
        metavar="REDUCER=PATH",
        help=(
            "one reducer's 2-D layout. Named rather than positional for the reason "
            "`coregister --embedding` is: indexing a labeled thing by position is "
            "the defect ADR 0007 exists to prevent."
        ),
    )
    parser.add_argument(
        "--cohort-report",
        default=None,
        help="cohort_report.json from download_pdbs, if this run produced one",
    )
    parser.add_argument(
        "--clusters",
        default=None,
        help=(
            "a two-column table of protid and cluster label. Supplying it turns on "
            "cross-cluster edge retention, which is the censoring number worth "
            "reading; without it the censoring section is bookkeeping."
        ),
    )
    return parser.parse_args()


def parse_named(pairs, flag: str) -> dict:
    out = {}
    for pair in pairs:
        key, separator, path = pair.partition("=")
        if not separator or not key.strip():
            raise SystemExit(
                f"{flag} expects NAME=PATH, got {pair!r}. A path containing an '=' is "
                "fine; the split is on the first one."
            )
        out[key.strip()] = path
    return out


def read_clusters(path: str) -> dict:
    """protid -> cluster label, read through the header rather than by position."""
    import pandas as pd

    frame = pd.read_csv(path, sep="\t")
    if frame.shape[1] < 2:
        raise SystemExit(f"{path} has {frame.shape[1]} column(s); need protid and a label.")
    if "protid" not in frame.columns:
        raise SystemExit(
            f"{path} has no 'protid' column (found {list(frame.columns)}). The cluster "
            "table is joined by label, never by row order."
        )
    label_column = next(c for c in frame.columns if c != "protid")
    return dict(zip(frame["protid"].astype(str), frame[label_column]))


def censoring_for_block(block, clusters=None):
    """The censoring report for one block, or a note saying why there is none.

    Only a *profile* block can answer this. Its features are the similarity
    matrix itself, rows and columns both being the block's protids in the same
    order, so the pairwise questions -- asymmetry, cross-cluster retention --
    are well posed. A block whose features are anything else has a censoring
    channel that is per cell of something other than a protein pair, and
    running the pairwise reports over it would produce numbers that look right
    and mean nothing.
    """
    from matrix_io import LabeledMatrix

    censored = block.channels.get("censored")
    if censored is None:
        return None
    values = block.features
    if (
        values is None
        or values.shape[0] != values.shape[1]
        or values.shape[0] != len(block.protids)
    ):
        shape = None if values is None else tuple(values.shape)
        return {
            "block_id": block.spec.id,
            "skipped": (
                "this block carries a censoring channel but its features are not a "
                f"square matrix over its own proteins (shape {shape} for "
                f"{len(block.protids)} proteins), so the pairwise censoring "
                "statistics are not defined for it."
            ),
        }
    from diagnostics.censoring import censoring_report

    matrix = LabeledMatrix(
        protids=list(block.protids),
        columns=list(block.protids),
        values=np.asarray(values),
        censored=np.asarray(censored, dtype=bool),
        source=f"block:{block.spec.id}",
    )
    scoped = None
    if clusters:
        # Only the proteins this block actually measured. A cluster table from
        # the legacy path covers the whole run, which is a superset whenever a
        # block dropped anything, and `cross_cluster_edge_retention` refuses a
        # partial assignment rather than guessing.
        missing = [p for p in block.protids if p not in clusters]
        if missing:
            print(
                f"[diagnose_space] {len(missing)} of block {block.spec.id!r}'s proteins "
                f"have no cluster assignment, so cross-cluster edge retention is not "
                f"reported for it: {missing[:5]}",
                file=sys.stderr,
            )
        else:
            scoped = {p: clusters[p] for p in block.protids}
    report = censoring_report(matrix, scoped)
    report["block_id"] = block.spec.id
    return report


def main() -> int:
    args = parse_args()
    config = from_legacy(load_config(args.configfile))

    if args.space_id not in config.spaces:
        raise SystemExit(
            f"space {args.space_id!r} is not defined. "
            f"Defined spaces: {config.space_ids() or '(none)'}"
        )
    space = config.spaces[args.space_id]
    store = BlockStore(args.output_dir)
    fused, index, blocks = features_for(space, store, config)
    protids = list(index)

    space_dir = os.path.join(args.output_dir, "spaces", args.space_id)
    os.makedirs(space_dir, exist_ok=True)

    report = {
        "space_id": args.space_id,
        "strategy": space.strategy,
        "blocks": list(space.blocks),
        "n_proteins": len(protids),
    }

    # 1. was the input measured
    clusters = read_clusters(args.clusters) if args.clusters else None
    censoring = [c for c in (censoring_for_block(b, clusters) for b in blocks) if c]
    if censoring:
        report["censoring"] = censoring

    # 2. was the cohort fair
    if args.cohort_report and os.path.exists(args.cohort_report):
        with open(args.cohort_report) as handle:
            report["cohort"] = json.load(handle)

    # 3. do the blocks say different things
    if len(blocks) > 1:
        aligned = {
            block.spec.id: index.align(
                block.protids,
                block.features,
                what=f"space {args.space_id!r} block {block.spec.id!r}",
            )
            for block in blocks
        }
        try:
            report["redundancy"] = redundancy(aligned).to_dict()
        except RedundancyError as error:
            raise SystemExit(f"space {args.space_id!r}: {error}") from error

    # 4. did the map survive two dimensions
    embeddings = parse_named(args.embedding, "--embedding")
    if embeddings:
        high = pairwise_distances(np.asarray(fused.values, dtype=np.float64))
        report["faithfulness"] = [
            _faithfulness_for(
                args.space_id, reducer, path, high, protids, space_dir, config.diagnostics.k
            )
            for reducer, path in sorted(embeddings.items())
        ]

    with open(os.path.join(space_dir, DIAGNOSTICS_FILENAME), "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=_plain)
        handle.write("\n")

    manifest = Manifest.build(
        "diagnostics",
        args.space_id,
        provider="diagnose_space",
        params={"space_id": args.space_id, "reducers": sorted(embeddings)},
        inputs={"block:" + block.spec.id: block.manifest.get("cache_key", "") for block in blocks},
        protids=protids,
        # The diagnostic sections this run actually produced, which is not the
        # same as the report's keys: `strategy` and `n_proteins` describe the
        # space rather than diagnosing it. A reader checking "was censoring
        # reported for this space" needs the difference.
        extra={"sections": sorted(set(report) & set(SECTIONS))},
    )
    manifest.write(os.path.join(space_dir, "manifest_diagnostics.json"))

    for note in _every_warning(report):
        print(f"[diagnose_space] {args.space_id}: {note}", file=sys.stderr)
    print(f"[diagnose_space] wrote {os.path.join(space_dir, DIAGNOSTICS_FILENAME)}")
    return 0


def _faithfulness_for(space_id, reducer, path, high, protids, space_dir, k) -> dict:
    """One reducer's layout, scored and written out per protein."""
    import pandas as pd

    frame = pd.read_csv(path, sep="\t", index_col=0)
    missing = [p for p in protids if p not in frame.index]
    if missing:
        raise SystemExit(
            f"{path} has no row for {len(missing)} of the space's {len(protids)} "
            f"proteins, starting with {missing[:5]}. The embedding and the blocks it "
            "came from disagree about the cohort."
        )
    low = pairwise_distances(frame.loc[list(protids)].to_numpy(dtype=np.float64))
    result = faithfulness(space_id, reducer, high, low, protids, k=k)
    result.to_frame().to_csv(os.path.join(space_dir, "faithfulness_" + reducer + ".tsv"), sep="\t")
    return result.to_dict()


def _every_warning(report) -> list:
    """Each section's warnings, flattened, so a reader of the log sees them all."""
    notes = []
    for section in ("censoring", "faithfulness"):
        for entry in report.get(section, []):
            notes.extend(entry.get("warnings", []) or entry.get("interpretation", []))
    for section in ("redundancy", "cohort"):
        notes.extend(report.get(section, {}).get("warnings", []))
    return notes


def _plain(value):
    """numpy scalars are not JSON, and a crashed rule is a worse diagnostic than
    a rounded one."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    raise SystemExit(main())
