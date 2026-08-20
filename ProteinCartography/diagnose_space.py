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

**Was the neighborhood ever determinate?** ``diagnostics/stability.py``. The one
question here that is not about a map: a protein whose k-th and (k+1)-th
neighbors are closer together than the measurement's own noise has a neighbor
list that would have come out differently on a repeat run, and no faithful
projection makes it mean anything.

**Is the partition a property of the proteins, or of the resolution knob?**
``diagnostics/partition.py``, over partitions from ``clustering.py``. Two
reports: a Leiden resolution sweep looking for a plateau, and negative controls
that say what the same numbers look like when there is nothing there. Both are
opt-in through `diagnostics.leiden_resolution_sweep` and
`diagnostics.negative_controls`, and both are skipped with a logged reason when
scanpy is absent -- a missing optional dependency is a reduced result, not an
error (ADR 0006 rule 2).

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
from diagnostics.partition import negative_controls, resolution_sweep
from diagnostics.redundancy import RedundancyError, redundancy
from diagnostics.stability import neighborhood_stability
from reduce_space import features_for
from spaces import layout
from spaces.manifest import Manifest
from spaces.store import BlockStore

DIAGNOSTICS_FILENAME = layout.DIAGNOSTICS_FILENAME

#: The four questions, in the order the module docstring asks them. A section is
#: absent when this run could not answer it -- a single-block space has no
#: redundancy, a cluster-mode run has no cohort report -- so the set that landed
#: is itself information and is recorded in the manifest.
SECTIONS = (
    "censoring",
    "cohort",
    "redundancy",
    "faithfulness",
    "stability",
    "resolution_sweep",
    "negative_controls",
)

#: Written beside ``diagnostics.json`` when this space could be clustered. Not
#: a declared snakemake output: whether it exists depends on scanpy being
#: importable and on the space having three proteins, and a rule that promises
#: a file it cannot always write fails the run instead of degrading it.
CLUSTERS_FILENAME = layout.CLUSTERS_FILENAME


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

    high = pairwise_distances(np.asarray(fused.values, dtype=np.float64))

    # 0. this space's own partition, which four of the sections below want and
    #    none of them had before group 8c. Computed once, written out, and
    #    preferred over the legacy one wherever a partition is needed.
    own = _own_partition(args.space_id, fused, protids, space_dir)
    legacy_clusters = read_clusters(args.clusters) if args.clusters else None
    if own is not None:
        report["partition"] = {
            "source": "this space",
            "resolution": own.resolution,
            "n_clusters": own.n_clusters,
            "n_neighbors": own.n_neighbors,
            "n_pcs": own.n_pcs,
            "seed": own.seed,
        }
    elif legacy_clusters:
        # FOLLOWUPS #41: this is the legacy *structural* partition, which is the
        # right one for the `structure` space and the wrong one for
        # `physicochemistry`. Recorded rather than silently substituted.
        report["partition"] = {
            "source": "the legacy structural clustering",
            "n_clusters": len(set(legacy_clusters.values())),
            "caveat": (
                "this space was not clustered in its own right, so every partition-"
                "dependent number below describes the structural clustering applied "
                "to this space's distances rather than this space's own grouping."
            ),
        }

    clusters = own.as_mapping() if own is not None else legacy_clusters

    # 1. was the input measured
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
            # Skip the section rather than killing the rule. Every other section
            # in this file skips with a logged reason when it cannot be
            # computed, and this one is not special: a two-protein space has an
            # undefined correlation, which is a missing paragraph, not a failed
            # run. The explorer depends on diagnostics, so raising here took the
            # whole run for one absent number.
            report["redundancy"] = {"skipped": str(error)}
            _skip(f"{args.space_id}: redundancy not computed ({error})")

    # 4. did the map survive two dimensions
    embeddings = parse_named(args.embedding, "--embedding")
    if embeddings:
        report["faithfulness"] = [
            _faithfulness_for(
                args.space_id, reducer, path, high, protids, space_dir, config.diagnostics.k
            )
            for reducer, path in sorted(embeddings.items())
        ]

    # 5. was the neighborhood ever determinate. Spelled `config.diagnostics.x`
    #    at each use rather than through a local alias, so that
    #    `test_diagnostics_config` can see the read: it looks for the attribute
    #    access itself, and a local name defeats that.
    if config.diagnostics.bootstrap_replicates:
        stability = neighborhood_stability(
            args.space_id,
            high,
            protids,
            k=config.diagnostics.k,
            replicates=config.diagnostics.bootstrap_replicates,
            subsample_fraction=config.diagnostics.subsample_fraction,
        )
        # The per-protein series, written beside the summary exactly as
        # faithfulness already writes its own. `to_dict` keeps the mean, the min
        # and the coin-flip LIST, which answers "is this space stable" and
        # discards the ramp that answers "is THIS protein stable" -- and the
        # source calls the per-protein overlay not optional (3.01).
        stability.to_frame().to_csv(
            os.path.join(space_dir, layout.stability_filename()), sep="\t"
        )
        report["stability"] = [stability.to_dict()]
    else:
        _skip(f"{args.space_id}: bootstrap_replicates is 0, so stability was not measured")

    # 6. is the partition a property of the proteins, or of the resolution knob
    if config.diagnostics.leiden_resolution_sweep:
        swept = _sweep_for(fused, protids, config.diagnostics.leiden_resolution_sweep)
        if swept is not None:
            report["resolution_sweep"] = resolution_sweep(args.space_id, high, swept).to_dict()

    if config.diagnostics.negative_controls:
        controls = _controls_for(
            args.space_id, high, protids, clusters, config.diagnostics.negative_controls
        )
        if controls is not None:
            report["negative_controls"] = controls.to_dict()

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
    manifest.write(
        os.path.join(space_dir, layout.manifest_filename(layout.DIAGNOSTICS_MANIFEST_KEY))
    )

    for note in _every_warning(report):
        print(f"[diagnose_space] {args.space_id}: {note}", file=sys.stderr)
    print(f"[diagnose_space] wrote {os.path.join(space_dir, DIAGNOSTICS_FILENAME)}")
    return 0


def _skip(reason: str) -> None:
    print(f"[diagnose_space] {reason}", file=sys.stderr)


def _own_partition(space_id, fused, protids, space_dir):
    """Cluster this space, or say in the log why it was not clustered.

    Returns ``None`` rather than raising. Every caller has a fallback, and a
    space that cannot be clustered should lose the partition-dependent
    diagnostics, not the run -- ADR 0006 rule 2's "a missing optional
    dependency is a reduced result, never an error", applied to the one
    dependency this work reaches for that it did not add.
    """
    from clustering import ClusteringError, is_available, leiden_partition

    available, explanation = is_available()
    if not available:
        _skip(f"{space_id}: not clustered ({explanation}); partition-dependent sections skipped")
        return None
    if len(protids) < 3:
        _skip(f"{space_id}: {len(protids)} proteins is too few to cluster")
        return None
    try:
        partition = leiden_partition(np.asarray(fused.values, dtype=np.float64), protids)
    except ClusteringError as error:
        _skip(f"{space_id}: not clustered ({error})")
        return None
    partition.to_frame().to_csv(os.path.join(space_dir, CLUSTERS_FILENAME), sep="\t")
    return partition


def _sweep_for(fused, protids, resolutions):
    """``{resolution: labels}``, or None when this space cannot be clustered."""
    from clustering import ClusteringError, is_available, sweep_resolutions

    available, explanation = is_available()
    if not available or len(protids) < 3:
        return None
    try:
        return sweep_resolutions(np.asarray(fused.values, dtype=np.float64), protids, resolutions)
    except ClusteringError as error:
        _skip(f"resolution sweep skipped: {error}")
        return None


def _controls_for(space_id, high, protids, clusters, requested):
    """The negative controls this config asked for, over this space's partition.

    ``shuffled_labels`` is always computed by ``diagnostics.partition`` and is
    listed in the config only so that asking for nothing means running nothing.
    ``random_distances`` needs a clusterer, and it is the one worth waiting
    for: fitting the pipeline's own Leiden to a matrix with no structure in it
    returns clusters with a positive silhouette, which is the number a reader
    has to see beside a real one.
    """
    if not clusters:
        _skip(f"{space_id}: no partition, so the negative controls were skipped")
        return None
    labels = [clusters.get(p) for p in protids]
    if any(label is None for label in labels) or len(set(labels)) < 2:
        _skip(
            f"{space_id}: the partition covers {len(set(labels) - {None})} cluster(s) over "
            f"{sum(label is not None for label in labels)} of {len(protids)} proteins, "
            "which the silhouette is not defined for; negative controls skipped"
        )
        return None

    extra, skipped = [], {}
    if "random_distances" in requested:
        extra, skipped = _random_distance_control(protids, high)
    return negative_controls(space_id, high, labels, extra=extra, skipped=skipped)


def _random_distance_control(protids, high):
    """Cluster a random matrix of the same shape and score the result.

    The random features are drawn at the observed data's scale so that the
    comparison is of structure rather than of units, and the seed is fixed so
    that a control which happens to look alarming can be reproduced.
    """
    from clustering import ClusteringError, is_available, leiden_partition

    available, explanation = is_available()
    if not available:
        return [], {"random_distances": explanation}
    if len(protids) < 3:
        return [], {"random_distances": f"{len(protids)} proteins is too few to cluster"}
    rng = np.random.RandomState(0)
    scale = float(np.median(high[~np.eye(len(protids), dtype=bool)])) or 1.0
    values = rng.normal(0.0, scale, size=(len(protids), min(len(protids), 32)))
    try:
        fitted = leiden_partition(values, protids)
    except ClusteringError as error:
        _skip(f"random-distance control skipped: {error}")
        return [], {"random_distances": str(error)}
    if fitted.n_clusters < 2:
        reason = (
            "the random matrix produced a single cluster, so it has no silhouette. At "
            "small N this happens for some spaces and not others -- it is a property of "
            "the cohort size, not of the space."
        )
        _skip(f"random-distance control skipped: {reason}")
        return [], {"random_distances": reason}
    return (
        [
            (
                "random_distances",
                "the same clustering fitted to a random matrix of the same shape and scale",
                pairwise_distances(values),
                fitted.labels,
            )
        ],
        {},
    )


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
    result.to_frame().to_csv(
        os.path.join(space_dir, layout.faithfulness_filename(reducer)), sep="\t"
    )
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
